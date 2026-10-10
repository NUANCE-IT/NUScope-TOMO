"""Sequential slice-to-slice registration for FIB-SEM stacks.

FIB-SEM slice series drift slowly (stage/beam drift between mills), so we
register each slice to its immediate predecessor with subpixel phase
cross-correlation and accumulate the shifts, rather than registering every
slice to a single fixed reference (which fails once drift exceeds the
correlation window). Everything reads/writes one slice at a time so this
scales to stacks that don't fit in RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.ndimage as ndi
import zarr
from skimage.registration import phase_cross_correlation

from ..data.volume import VolumeData


def estimate_shifts(
    volume: VolumeData,
    upsample_factor: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Return cumulative (dy, dx) shifts, shape (Z, 2), that register every
    slice into slice 0's coordinate frame.

    Computed slice-pair by slice-pair (constant memory) via subpixel phase
    cross-correlation, then accumulated.
    """
    n = volume.shape[0]
    shifts = np.zeros((n, 2), dtype=np.float64)
    if n < 2:
        return shifts

    prev = volume.get_slice(0, 0).astype(np.float32)
    cumulative = np.zeros(2, dtype=np.float64)
    if progress_callback:
        progress_callback(1, n)

    for i in range(1, n):
        cur = volume.get_slice(0, i).astype(np.float32)
        pair_shift, _error, _diffphase = phase_cross_correlation(
            prev, cur, upsample_factor=upsample_factor
        )
        cumulative = cumulative + pair_shift
        shifts[i] = cumulative
        prev = cur
        if progress_callback:
            progress_callback(i + 1, n)

    return shifts


def valid_crop_bounds(shifts: np.ndarray, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Y/X slices that stay inside every shifted frame (drops the border
    that padding introduces once slices have been shifted into a common
    frame)."""
    h, w = shape
    dy = shifts[:, 0]
    dx = shifts[:, 1]
    top = int(np.ceil(max(0.0, dy.max())))
    bottom = int(np.floor(min(0.0, dy.min())))
    left = int(np.ceil(max(0.0, dx.max())))
    right = int(np.floor(min(0.0, dx.min())))
    return (
        slice(top, h + bottom),
        slice(left, w + right),
    )


def apply_shifts(
    volume: VolumeData,
    shifts: np.ndarray,
    output_zarr_path: str | Path,
    crop_to_valid: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VolumeData:
    """Write an aligned copy of `volume` to a new zarr store, one slice at a
    time. Optionally crops away the border introduced by shifting so the
    result has no zero-padded edges.
    """
    output_zarr_path = Path(output_zarr_path)
    n, h, w = volume.shape

    y_sl, x_sl = valid_crop_bounds(shifts, (h, w)) if crop_to_valid else (slice(0, h), slice(0, w))
    out_h = y_sl.stop - y_sl.start
    out_w = x_sl.stop - x_sl.start

    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(
        str(output_zarr_path),
        mode="w",
        shape=(n, out_h, out_w),
        chunks=(min(8, n), out_h, out_w),
        dtype=volume.dtype,
    )

    for i in range(n):
        raw = volume.get_slice(0, i)
        shifted = ndi.shift(raw, shift=shifts[i], order=1, mode="constant", cval=0)
        store[i] = shifted[y_sl, x_sl].astype(volume.dtype)
        if progress_callback:
            progress_callback(i + 1, n)

    import dask.array as da
    import json

    meta = {
        "voxel_size": volume.voxel_size.__dict__,
        "name": f"{volume.name}_aligned",
        "alignment_shifts": shifts.tolist(),
    }
    (output_zarr_path / VolumeData.METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    array = da.from_zarr(store)
    return VolumeData(
        array,
        store_path=output_zarr_path,
        voxel_size=volume.voxel_size,
        name=f"{volume.name}_aligned",
    )


def run_alignment(
    volume: VolumeData,
    output_zarr_path: str | Path,
    upsample_factor: int = 10,
    crop_to_valid: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VolumeData:
    """Estimate shifts then apply them, reporting combined progress over
    both passes (0..2N) through a single callback — convenient for driving
    one Qt progress dialog across the whole two-phase operation."""
    n = volume.shape[0]

    def phase1(done: int, _total: int) -> None:
        if progress_callback:
            progress_callback(done, 2 * n)

    shifts = estimate_shifts(volume, upsample_factor=upsample_factor, progress_callback=phase1)

    def phase2(done: int, _total: int) -> None:
        if progress_callback:
            progress_callback(n + done, 2 * n)

    return apply_shifts(
        volume, shifts, output_zarr_path, crop_to_valid=crop_to_valid, progress_callback=phase2
    )
