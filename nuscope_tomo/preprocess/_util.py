"""Helpers for slice-wise (per-FIB-slice) processing of out-of-core volumes."""
from __future__ import annotations

from typing import Callable

import dask.array as da
import numpy as np

from ..core.volume import Volume


def slicewise(volume: Volume, func: Callable[[np.ndarray], np.ndarray],
              dtype=None, z_chunk: int = 8) -> da.Array:
    """Apply ``func`` to every 2D slice lazily (full XY per chunk)."""
    arr = volume.data.rechunk((z_chunk, -1, -1))
    dtype = dtype or volume.dtype

    def block(b):
        return np.stack([func(s) for s in b]).astype(dtype, copy=False)

    return arr.map_blocks(block, dtype=dtype)


def to_dtype(img: np.ndarray, dtype, lo=None, hi=None) -> np.ndarray:
    """Clip/round float results back to the source dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
        return np.clip(np.rint(img), lo, hi).astype(dtype)
    return img.astype(dtype)


def sample_slices(volume: Volume, n: int = 24) -> np.ndarray:
    idx = np.unique(np.linspace(0, volume.shape[0] - 1, min(n, volume.shape[0])).astype(int))
    return np.stack([volume.get_slice(0, i) for i in idx]), idx
