"""Thresholding and labeling.

Thresholding is elementwise, so it stays out-of-core: it is applied chunk by
chunk straight from the on-disk zarr store into a new uint8 mask store,
never materializing more than one chunk in RAM. Connected-component
labeling and small-object cleanup are not embarrassingly chunk-local
(a component can span a chunk boundary), so those operate on an in-memory
array — call them on `VolumeData.preview()` for a whole-volume look or
`VolumeData.roi(...)` for a full-resolution region of interest.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import dask.array as da
import numpy as np
import zarr
from skimage.filters import threshold_otsu
from skimage.measure import label as _label
from skimage.morphology import remove_small_objects

from ..data.volume import VolumeData


def otsu_threshold(volume: VolumeData, bins: int = 256, sample_step: int = 4) -> float:
    """Otsu threshold computed from an out-of-core histogram (subsampled)."""
    hist, edges = volume.histogram(bins=bins, sample_step=sample_step)
    centers = (edges[:-1] + edges[1:]) / 2
    return float(threshold_otsu(hist=(hist, centers)))


def apply_threshold(
    volume: VolumeData,
    low: float,
    high: float,
    output_zarr_path: str | Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VolumeData:
    """Write a uint8 {0,1} mask (low <= voxel <= high) to a new zarr store,
    processed one z-chunk at a time so peak memory is O(chunk), not O(volume).
    """
    output_zarr_path = Path(output_zarr_path)
    n, h, w = volume.shape
    chunk_z = volume.chunks[0][0] if volume.chunks[0] else 8

    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(
        str(output_zarr_path),
        mode="w",
        shape=(n, h, w),
        chunks=(chunk_z, h, w),
        dtype=np.uint8,
    )

    total_chunks = math.ceil(n / chunk_z)
    done = 0
    z = 0
    while z < n:
        end = min(z + chunk_z, n)
        block = volume.roi(slice(z, end), slice(None), slice(None))
        store[z:end] = ((block >= low) & (block <= high)).astype(np.uint8)
        z = end
        done += 1
        if progress_callback:
            progress_callback(done, total_chunks)

    meta = {
        "voxel_size": volume.voxel_size.__dict__,
        "name": f"{volume.name}_mask",
        "threshold_low": low,
        "threshold_high": high,
    }
    (output_zarr_path / VolumeData.METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    array = da.from_zarr(store)
    return VolumeData(
        array,
        store_path=output_zarr_path,
        voxel_size=volume.voxel_size,
        name=f"{volume.name}_mask",
    )


def label_components(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    """Connected-component label an in-memory binary mask, dropping objects
    of `min_size` voxels or fewer. Operates in memory — run on a preview
    or ROI, not a full out-of-core volume."""
    cleaned = remove_small_objects(mask.astype(bool), max_size=min_size)
    return _label(cleaned)
