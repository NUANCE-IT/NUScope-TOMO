"""From class maps to objects: cleanup, connected components and
watershed separation of touching particles. Distances use the physical voxel
size, so anisotropic FIB-SEM voxels (thick slices) are handled correctly.
These operate in memory; run them on a binned volume or an ROI for huge stacks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed

from ..core.registry import op
from ..core.volume import Volume


def _ball(volume: Volume, radius_nm: float) -> np.ndarray:
    r = [max(0, int(round(radius_nm / v))) for v in volume.voxel_size.as_tuple()]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return ((zz / max(r[0], 1e-9)) ** 2 * (r[0] > 0) + (yy / max(r[1], 1)) ** 2
            + (xx / max(r[2], 1)) ** 2) <= 1.0


@op("segment.smooth_labels")
def smooth_labels(volume: Volume, size: int = 3) -> Volume:
    """Majority (mode) filter on a class map: removes salt-and-pepper voxels."""
    lab = volume.numpy()
    classes = np.unique(lab)
    votes = np.stack([ndi.uniform_filter((lab == c).astype(np.float32), size) for c in classes])
    return volume.derive(classes[np.argmax(votes, 0)].astype(lab.dtype))


@op("segment.extract_class")
def extract_class(volume: Volume, label: int = 1, min_size_nm3: float = 0.0,
                  fill_holes_nm3: float = 0.0, open_nm: float = 0.0) -> Volume:
    """Binary mask of one class with optional opening / small-object cleanup."""
    m = volume.numpy() == label
    if open_nm:
        m = ndi.binary_opening(m, structure=_ball(volume, open_nm))
    vv = volume.voxel_size.voxel_volume_nm3
    if min_size_nm3:
        m = remove_small_objects(m, max_size=int(min_size_nm3 / vv))
    if fill_holes_nm3:
        m = remove_small_holes(m, max_size=int(fill_holes_nm3 / vv))
    name = volume.meta.get("classes", {}).get(str(label), f"class_{label}")
    return volume.derive(m.astype(np.uint8), name=name, kind="labels", classes={"1": name})


@op("segment.connected_components")
def connected_components(volume: Volume, min_size_nm3: float = 0.0,
                         connectivity: int = 1) -> Volume:
    """Label connected objects of a binary mask (uint32 object ids)."""
    m = volume.numpy() > 0
    st = ndi.generate_binary_structure(3, connectivity)
    lab, n = ndi.label(m, structure=st)
    if min_size_nm3:
        lab = remove_small_objects(lab, max_size=int(min_size_nm3 / volume.voxel_size.voxel_volume_nm3))
        lab, _, _ = _relabel(lab)
    return volume.derive(lab.astype(np.uint32), kind="labels", objects=True,
                         n_objects=int(lab.max()))


@op("segment.separate")
def separate(volume: Volume, min_distance_nm: float = 100.0, h_nm: float = 15.0,
             min_size_nm3: float = 0.0, compactness: float = 0.0) -> Volume:
    """Split touching particles: anisotropic distance transform + h-maxima seeds + watershed."""
    m = volume.numpy() > 0
    dist = ndi.distance_transform_edt(m, sampling=volume.voxel_size.as_tuple())
    # smooth the distance map a little so shallow ripples don't seed objects
    sig = [min_distance_nm / 4 / v for v in volume.voxel_size.as_tuple()]
    ds = ndi.gaussian_filter(dist, [min(s, 2) for s in sig])
    from skimage.morphology import h_maxima
    fp = _ball(volume, min_distance_nm / 2)
    peaks = h_maxima(ds, h_nm, footprint=fp) & m
    markers, _ = ndi.label(peaks)
    lab = watershed(-ds, markers, mask=m, compactness=compactness)
    if min_size_nm3:
        lab = remove_small_objects(lab, max_size=int(min_size_nm3 / volume.voxel_size.voxel_volume_nm3))
    lab, _, _ = _relabel(lab)
    return volume.derive(lab.astype(np.uint32), kind="labels", objects=True,
                         n_objects=int(lab.max()))


def _relabel(lab: np.ndarray):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(lab)


@op("segment.remove_border_objects")
def remove_border_objects(volume: Volume, axes: Optional[list] = None) -> Volume:
    """Drop objects touching the volume boundary (biased/truncated for statistics)."""
    lab = volume.numpy()
    axes = axes or [0, 1, 2]
    border = set()
    for ax in axes:
        for idx in (0, -1):
            border |= set(np.unique(np.take(lab, idx, axis=ax)).tolist())
    border.discard(0)
    out = np.where(np.isin(lab, list(border)), 0, lab)
    out, _, _ = _relabel(out)
    return volume.derive(out.astype(lab.dtype), n_objects=int(out.max()))


__all__ = ["smooth_labels", "extract_class", "connected_components", "separate",
           "remove_border_objects"]
