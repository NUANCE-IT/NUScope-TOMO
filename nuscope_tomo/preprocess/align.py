"""Slice-to-slice drift correction for FIB-SEM stacks.

FIB-SEM drift is sequential (each slice is imaged after milling the
previous one), so each slice is registered to its predecessor with sub-pixel
phase correlation and the shifts are accumulated. Single-slice failures
(charging, a missing frame) are caught with a robust outlier test before
accumulation so they do not propagate down the stack. Optionally a smooth
trend can be removed, so a genuinely tilted feature is not "straightened".

Caveat (important for thick slices): correlation cannot tell stage drift
from microstructure that is *inclined* to the milling direction. With e.g.
50 nm slices and 5 nm pixels, a 45° grain boundary moves 10 px per slice and
looks like drift. Hence: ``roi`` restricts the measurement to a fiducial
(a milled mark, the cap edge, the trench wall), ``highpass_slices`` keeps
only abrupt drift events, and stacks that are already registered (noise-
level steps, e.g. exported "registered" from the acquisition software) are
detected and left untouched.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from ..core.registry import op
from ..core.volume import Volume
from ._util import to_dtype


def _prep(img: np.ndarray, patch: int, roi=None) -> np.ndarray:
    if roi is not None:
        (y0, y1), (x0, x1) = roi
        img = img[y0:y1, x0:x1]
    h, w = img.shape
    ph, pw = min(patch, h), min(patch, w)
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    a = img[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    # gradient magnitude is robust to slow brightness changes between slices
    a = ndi.gaussian_filter(a, 1.0)
    g = np.hypot(ndi.sobel(a, 0), ndi.sobel(a, 1))
    win = np.outer(np.hanning(ph), np.hanning(pw)).astype(np.float32)
    return (g - g.mean()) * win


def estimate_drift(volume: Volume, patch: int = 1024, upsample: int = 10,
                   offsets: Sequence[int] = (1, 2, 4, 8), outlier_mad: float = 5.0,
                   roi: Optional[Sequence[Sequence[int]]] = None, workers: int = 8,
                   progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """Absolute per-slice (dy, dx) positions that register the stack.

    Pairwise shifts are measured between each slice and several
    predecessors (``offsets``) and the positions are solved jointly by
    least squares. With offset 1 only this is plain sequential accumulation,
    whose sub-pixel noise random-walks over hundreds of slices; the longer
    baselines pin that walk down. Outlier pairs (charging, missing frames)
    are rejected robustly before the solve.
    """
    n = volume.shape[0]
    cache: dict[int, np.ndarray] = {}

    def prep(i):
        if i not in cache:
            cache[i] = _prep(volume.get_slice(0, i), patch, roi)
        return cache[i]

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(prep, range(n)))
    pairs = [(i - o, i) for o in offsets for i in range(o, n)]

    def measure(p):
        a, b = p
        s, err, _ = phase_cross_correlation(cache[a], cache[b], upsample_factor=upsample,
                                            normalization=None)
        return s, err

    meas = np.zeros((len(pairs), 2))
    errs = np.zeros(len(pairs))
    with ThreadPoolExecutor(workers) as ex:
        for k, (s, err) in enumerate(ex.map(measure, pairs)):
            meas[k], errs[k] = s, err
            if progress and k % 64 == 0:
                progress(k, len(pairs))
    cache.clear()

    pairs_a = np.array(pairs)
    gap = pairs_a[:, 1] - pairs_a[:, 0]
    keep = np.ones(len(pairs), bool)
    for o in offsets:  # robust outlier rejection per baseline length
        sel = gap == o
        for ax in range(2):
            d = meas[sel, ax]
            med = ndi.median_filter(d, 9, mode="nearest")
            mad = 1.4826 * np.median(np.abs(d - med)) + 1e-3
            bad = np.abs(d - med) > outlier_mad * max(mad, 0.25)
            keep[np.flatnonzero(sel)[bad]] = False

    # least squares: p[b] - p[a] = shift(a->b), with p[0] = 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    rows = np.flatnonzero(keep)
    m = len(rows)
    A = coo_matrix((np.r_[np.ones(m), -np.ones(m), [1.0]],
                    (np.r_[np.arange(m), np.arange(m), [m]],
                     np.r_[pairs_a[rows, 1], pairs_a[rows, 0], [0]])), shape=(m + 1, n)).tocsr()
    pos = np.column_stack([lsqr(A, np.r_[meas[rows, ax], 0.0], atol=1e-10, btol=1e-10)[0]
                           for ax in range(2)])
    first = gap == 1
    step = np.zeros((n, 2))
    step[pairs_a[first, 1]] = meas[first]
    resid = np.zeros(n)
    resid[pairs_a[first, 1]] = errs[first]
    outl = np.zeros(n, bool)
    outl[pairs_a[first & ~keep, 1]] = True
    return pd.DataFrame({"z": np.arange(n), "dy_step": step[:, 0], "dx_step": step[:, 1],
                         "dy": pos[:, 0], "dx": pos[:, 1], "error": resid, "outlier": outl})


def _apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    if abs(dy - round(dy)) < 1e-3 and abs(dx - round(dx)) < 1e-3:
        out = np.zeros_like(img)
        dy, dx = int(round(dy)), int(round(dx))
        h, w = img.shape
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[yd, xd] = img[ys, xs]
        return out
    return to_dtype(ndi.shift(img.astype(np.float32), (dy, dx), order=1, cval=0), img.dtype)


def is_registered(drift: pd.DataFrame, step_px: float = 1.0) -> bool:
    """True if slice-to-slice steps are at noise level: robust spread below
    ``step_px`` and essentially no drift jumps (isolated outliers allowed)."""
    steps = drift[["dy_step", "dx_step"]].to_numpy()[1:]
    mad = 1.4826 * np.median(np.abs(steps - np.median(steps, 0)), 0)
    jumps = (np.abs(steps) > 3 * step_px).any(1).mean()
    return bool(mad.max() <= step_px and jumps < 0.01)


@op("preprocess.align")
def align(volume: Volume, method: str = "multibaseline", patch: int = 1024,
          upsample: int = 10, roi: Optional[Sequence[Sequence[int]]] = None,
          highpass_slices: float = 0.0, skip_if_registered: bool = True,
          subpixel: bool = True, crop_valid: bool = True, outlier_mad: float = 5.0) -> dict:
    """Register slices (sub-pixel phase correlation, robust drift model).

    method: "multibaseline" (least squares over gaps 1, 2, 4, 8) or
    "pairwise" (classic sequential accumulation).
    """
    offsets = (1, 2, 4, 8) if method == "multibaseline" else (1,)
    drift = estimate_drift(volume, patch=patch, upsample=upsample, offsets=offsets,
                           outlier_mad=outlier_mad, roi=roi)
    if skip_if_registered and roi is None and is_registered(drift):
        drift["dy_applied"] = drift["dx_applied"] = 0.0
        return {"volume": volume.derive(volume.data, aligned="already registered"),
                "drift": drift}
    dy, dx = drift["dy"].to_numpy(), drift["dx"].to_numpy()
    if highpass_slices:
        dy = dy - ndi.gaussian_filter1d(dy, highpass_slices, mode="nearest")
        dx = dx - ndi.gaussian_filter1d(dx, highpass_slices, mode="nearest")
    dy, dx = dy - np.median(dy), dx - np.median(dx)
    if not subpixel:
        dy, dx = np.round(dy), np.round(dx)
    drift["dy_applied"], drift["dx_applied"] = dy, dx

    arr = volume.data.rechunk((8, -1, -1))

    def block(b, block_info=None):
        z0 = block_info[0]["array-location"][0][0]
        return np.stack([_apply_shift(s, dy[z0 + k], dx[z0 + k]) for k, s in enumerate(b)])

    out = arr.map_blocks(block, dtype=volume.dtype)
    h, w = volume.shape[1:]
    if crop_valid:
        y0, y1 = int(np.ceil(max(dy.max(), 0))), int(np.floor(h + min(dy.min(), 0)))
        x0, x1 = int(np.ceil(max(dx.max(), 0))), int(np.floor(w + min(dx.min(), 0)))
        out = out[:, y0:y1, x0:x1]
    aligned = volume.derive(out, name=volume.name, aligned=True,
                            drift_px_range=[float(np.ptp(dy)), float(np.ptp(dx))])
    return {"volume": aligned, "drift": drift}


__all__ = ["estimate_drift", "align"]
