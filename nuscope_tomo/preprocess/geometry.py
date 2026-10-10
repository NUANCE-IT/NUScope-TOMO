"""Geometric pre-processing: ROI detection/cropping, binning, resampling and
the cross-section tilt correction of slice-and-view imaging."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from ..core.registry import op
from ..core.volume import Volume
from ._util import sample_slices, slicewise, to_dtype


def detect_sample_roi(volume: Volume, n_samples: int = 24, margin: int = 20,
                      edge_frac: float = 0.5) -> dict:
    """Locate the milled block face in a slice-and-view stack.

    Excludes (1) the dark trench side walls / zero padding left by alignment,
    (2) the bright protective Pt cap at the top whose interface moves with z.
    Returns ``{"y": (y0, y1), "x": (x0, x1)}``.
    """
    st, _ = sample_slices(volume, n_samples)
    st = st.astype(np.float32)
    mn = st.min(0)
    h, w = mn.shape

    # columns: the sample is the longest bright run of the min-projection
    core = mn[h // 4: 3 * h // 4]
    col = ndi.median_filter(np.median(core, 0), 9)
    level = np.percentile(col, 90)
    x0, x1 = _longest_run(col > edge_frac * level)
    # rows: valid where the min-projection is non-empty
    row = ndi.median_filter(np.median(mn[:, x0:x1], 1), 9)
    y_valid = row > 0.3 * np.median(row)
    y_lo, y_hi = _longest_run(y_valid)

    # Pt cap: bright band near the top; find its lower edge per column/slice
    sm = ndi.uniform_filter1d(st[:, :, x0:x1:8], 15, axis=1)
    top = h // 2
    sample_level = np.median(sm[:, h // 2:])
    cap_level = np.percentile(sm[:, :top], 97)
    if cap_level < 1.4 * sample_level:  # no bright cap in view
        cap_level = np.inf
    cap_thr = 0.5 * (sample_level + cap_level)
    bright = sm[:, :top] > cap_thr
    # cap = the first bright run that starts in the top quarter (bright
    # particles further down are not the cap)
    edges = []
    for prof in bright.transpose(0, 2, 1).reshape(-1, top):
        if not prof[: top // 2].any():
            continue
        f = int(np.argmax(prof))
        rest = ~prof[f:]
        edges.append(f + (int(np.argmax(rest)) if rest.any() else top - f))
    cap_edge = int(np.percentile(edges, 98)) if edges else 0
    y0 = max(y_lo, cap_edge) + margin
    return {"y": (int(y0), int(y_hi - margin)), "x": (int(x0 + margin), int(x1 - margin))}


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    lab, n = ndi.label(mask)
    if n == 0:
        return 0, len(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    idx = np.flatnonzero(lab == k)
    return int(idx[0]), int(idx[-1] + 1)


@op("preprocess.auto_crop")
def auto_crop(volume: Volume, margin: int = 20, edge_frac: float = 0.5) -> Volume:
    """Crop to the block face: remove Pt cap, trench walls and padding."""
    roi = detect_sample_roi(volume, margin=margin, edge_frac=edge_frac)
    (y0, y1), (x0, x1) = roi["y"], roi["x"]
    return volume.derive(volume.data[:, y0:y1, x0:x1], name=f"{volume.name}",
                         crop={"y": [y0, y1], "x": [x0, x1]})


@op("preprocess.crop")
def crop(volume: Volume, z: Optional[Sequence[int]] = None, y: Optional[Sequence[int]] = None,
         x: Optional[Sequence[int]] = None) -> Volume:
    """Crop to explicit index ranges [start, stop) per axis."""
    sl = tuple(slice(*r) if r else slice(None) for r in (z, y, x))
    return volume.derive(volume.data[sl], crop={"z": z, "y": y, "x": x})


@op("preprocess.bin")
def bin_volume(volume: Volume, factors: Sequence[int] = (1, 2, 2)) -> Volume:
    """Block-average binning (noise reduction + size reduction)."""
    f = dict(enumerate(int(v) for v in factors))
    trimmed = volume.data[tuple(slice(0, s // f[i] * f[i]) for i, s in enumerate(volume.shape))]
    out = da.coarsen(np.mean, trimmed.astype(np.float32), f, trim_excess=True)
    if np.issubdtype(volume.dtype, np.integer):
        out = da.round(out).astype(volume.dtype)
    return volume.derive(out, voxel_size=volume.voxel_size.scaled(*factors))


@op("preprocess.resample_isotropic")
def resample_isotropic(volume: Volume, voxel_nm: Optional[float] = None, order: int = 1) -> Volume:
    """Resample to cubic voxels (default: the coarsest axis spacing)."""
    vs = np.array(volume.voxel_size.as_tuple())
    target = float(voxel_nm or vs.max())
    zoom = vs / target
    arr = volume.numpy().astype(np.float32)
    out = ndi.zoom(arr, zoom, order=order, prefilter=order > 1)
    out = to_dtype(out, volume.dtype) if volume.kind != "labels" else out.astype(volume.dtype)
    from ..core.volume import VoxelSize
    return volume.derive(out, voxel_size=VoxelSize(target, target, target))


@op("preprocess.tilt_correct")
def tilt_correct(volume: Volume, stage_tilt_deg: float = 52.0) -> Volume:
    """Correct foreshortening of the cross-section imaged at the SEM stage
    tilt: the Y pixel size becomes ``px / sin(tilt)``. Only calibration is
    changed (no resampling), so it is exact and reversible."""
    s = 1.0 / math.sin(math.radians(stage_tilt_deg))
    return volume.derive(volume.data, voxel_size=volume.voxel_size.scaled(1, s, 1),
                         tilt_correction=stage_tilt_deg)


__all__ = ["detect_sample_roi", "auto_crop", "crop", "bin_volume", "resample_isotropic",
           "tilt_correct", "slicewise"]
