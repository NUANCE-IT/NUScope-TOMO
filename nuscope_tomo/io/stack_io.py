"""Import a folder of 2D slice images (FIB-SEM SE images) into a zarr-backed
VolumeData, one slice at a time so peak memory stays at a single frame
regardless of how many slices there are.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import dask.array as da
import numpy as np
import tifffile
import zarr

from ..data.volume import VolumeData, VoxelSize

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}

_TRAILING_NUMBER = re.compile(r"(\d+)(?=\D*$)")


def _natural_key(path: Path):
    """Sort key that orders 'slice2' before 'slice10'."""
    match = _TRAILING_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else -1, path.stem)


def list_slice_files(folder: str | Path, pattern: Optional[str] = None) -> list[Path]:
    """Return image files in `folder`, naturally sorted by trailing number.

    `pattern` is an optional glob (e.g. "*.tiff") to restrict which files
    are considered; by default every recognized image suffix is included.
    """
    folder = Path(folder)
    if pattern:
        candidates = list(folder.glob(pattern))
    else:
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not candidates:
        raise FileNotFoundError(f"No image files found in {folder}")
    return sorted(candidates, key=_natural_key)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    else:
        import imageio.v3 as iio

        arr = iio.imread(str(path))
    if arr.ndim == 3:
        # Grayscale-as-RGB SEM export: collapse identical channels.
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[..., 0]
    return arr


def import_image_stack(
    folder: str | Path,
    output_zarr_path: str | Path,
    voxel_size: Optional[VoxelSize] = None,
    name: Optional[str] = None,
    pattern: Optional[str] = None,
    chunk_slices: int = 8,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VolumeData:
    """Read a natural-sorted image sequence from `folder` into a new zarr
    store at `output_zarr_path`, writing one slice at a time.

    `progress_callback(done, total)` is called after each slice is written,
    if given (used to drive a Qt progress dialog).
    """
    folder = Path(folder)
    output_zarr_path = Path(output_zarr_path)
    files = list_slice_files(folder, pattern=pattern)
    n = len(files)

    first = _read_image(files[0])
    h, w = first.shape[:2]
    dtype = first.dtype

    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(
        str(output_zarr_path),
        mode="w",
        shape=(n, h, w),
        chunks=(chunk_slices, h, w),
        dtype=dtype,
    )

    store[0] = first
    if progress_callback:
        progress_callback(1, n)

    for i, path in enumerate(files[1:], start=1):
        img = _read_image(path)
        if img.shape[:2] != (h, w):
            raise ValueError(
                f"Slice {path.name} has shape {img.shape[:2]}, expected {(h, w)}"
            )
        store[i] = img
        if progress_callback:
            progress_callback(i + 1, n)

    meta = {
        "voxel_size": (voxel_size or VoxelSize()).__dict__,
        "name": name or folder.name,
        "source_folder": str(folder),
        "source_files": [f.name for f in files],
    }
    import json

    (output_zarr_path / VolumeData.METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    array = da.from_zarr(store)
    return VolumeData(
        array,
        store_path=output_zarr_path,
        voxel_size=voxel_size or VoxelSize(),
        name=name or folder.name,
    )
