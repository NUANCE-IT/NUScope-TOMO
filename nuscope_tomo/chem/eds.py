"""Correlative EDS chemical tomography for FIB slice-and-view.

During slice-and-view with EDS, every N-th milled face is also mapped by
EDS. The element maps are coarser (and noisier, Poisson counts) than the
SE/BSE images. This module

1. imports per-slice element maps (one folder per element, or one folder
   per slice) into 3D element volumes,
2. maps them onto the SE grid (field-of-view scaling + sub-pixel
   registration against the SE image),
3. denoises counts in 3D and converts to relative composition,
4. finds chemical phases without supervision (k-means / Gaussian mixture on
   composition vectors, or NMF end-members),
5. fuses chemistry with the SE segmentation: composition per SE class or
   per object, and EDS channels as extra classifier features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from ..core.registry import op
from ..core.volume import Volume, VoxelSize
from ..io.stack import list_slices, read_image


class EDSStack:
    """A set of co-registered 3D element-count volumes (same grid)."""

    def __init__(self, maps: dict[str, np.ndarray], voxel_size: VoxelSize,
                 z_index: Optional[np.ndarray] = None):
        self.maps = {k: np.asarray(v, np.float32) for k, v in maps.items()}
        self.voxel_size = voxel_size
        shapes = {v.shape for v in self.maps.values()}
        if len(shapes) != 1:
            raise ValueError(f"Element maps differ in shape: {shapes}")
        self.shape = shapes.pop()
        # which SE slice each EDS slice corresponds to
        self.z_index = np.arange(self.shape[0]) if z_index is None else np.asarray(z_index)

    @property
    def elements(self) -> list[str]:
        return list(self.maps)

    def volume(self, element: str) -> Volume:
        return Volume(self.maps[element], self.voxel_size, element, "element",
                      {"element": element})

    def stacked(self) -> np.ndarray:
        return np.stack([self.maps[e] for e in self.elements], -1)

    def __repr__(self):
        return f"EDSStack({self.elements}, shape={self.shape}, voxel={self.voxel_size.as_tuple()})"


def read_eds_folders(root: str | Path, voxel_size: VoxelSize,
                     elements: Optional[Sequence[str]] = None,
                     z_index: Optional[Sequence[int]] = None) -> EDSStack:
    """Read ``root/<Element>/*.tif`` (one image per EDS slice) for each element."""
    root = Path(root)
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if elements:
        dirs = [d for d in dirs if d.name in elements]
    maps = {d.name: np.stack([read_image(f).astype(np.float32) for f in list_slices(d)])
            for d in dirs}
    return EDSStack(maps, voxel_size, z_index)


# --------------------------------------------------------------- registration
def register_to_se(eds: EDSStack, se: Volume, reference: Optional[str] = None,
                   fov_px: Optional[Sequence[int]] = None, upsample: int = 20,
                   order: int = 1) -> EDSStack:
    """Resample EDS maps onto the SE pixel grid and register them.

    ``fov_px`` = (y0, y1, x0, x1) of the EDS field of view in SE pixels
    (default: the full SE frame). The shift between SE and EDS is estimated
    on the ``reference`` element (default: the map best correlated with SE
    contrast) per EDS slice, then median-filtered along z for robustness.
    """
    ny, nx = se.shape[1:]
    y0, y1, x0, x1 = fov_px or (0, ny, 0, nx)
    ez, ey, ex = eds.shape
    zoom = ((y1 - y0) / ey, (x1 - x0) / ex)
    ref = reference or _best_reference(eds, se, zoom, (y0, y1, x0, x1))
    shifts = []
    for k, z in enumerate(eds.z_index):
        se_img = se.get_slice(0, int(z))[y0:y1, x0:x1].astype(np.float32)
        se_lo = _block_mean(se_img, ey, ex)
        e = ndi.gaussian_filter(eds.maps[ref][k], 1.0)
        a, b = _norm(se_lo), _norm(e)
        s1, err1, _ = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)
        s2, err2, _ = phase_cross_correlation(a, -b, upsample_factor=upsample, normalization=None)
        shifts.append(s1 if err1 <= err2 else s2)  # element may anti-correlate with SE
    shifts = ndi.median_filter(np.asarray(shifts), size=(5, 1), mode="nearest")
    out = {}
    for el, m in eds.maps.items():
        vol = np.zeros((ez, y1 - y0, x1 - x0), np.float32)
        for k in range(ez):
            sh = ndi.shift(m[k], shifts[k], order=1, mode="nearest")
            up = ndi.zoom(sh, zoom, order=order)
            vol[k, : up.shape[0], : up.shape[1]] = up[: y1 - y0, : x1 - x0]
        out[el] = vol
    vs = VoxelSize(eds.voxel_size.z, se.voxel_size.y, se.voxel_size.x)
    reg = EDSStack(out, vs, eds.z_index)
    reg.shifts_px = shifts
    reg.reference = ref
    reg.fov_px = (y0, y1, x0, x1)
    return reg


def _block_mean(img, ty, tx):
    return ndi.zoom(ndi.uniform_filter(img, (img.shape[0] / ty, img.shape[1] / tx)),
                    (ty / img.shape[0], tx / img.shape[1]), order=1)[:ty, :tx]


def _norm(a):
    a = a - a.mean()
    return a / (a.std() + 1e-6) * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))


def _best_reference(eds, se, zoom, fov):
    y0, y1, x0, x1 = fov
    k = eds.shape[0] // 2
    se_lo = _block_mean(se.get_slice(0, int(eds.z_index[k]))[y0:y1, x0:x1].astype(np.float32),
                        *eds.shape[1:])
    scores = {el: abs(np.corrcoef(ndi.gaussian_filter(m[k], 1).ravel(), se_lo.ravel())[0, 1])
              for el, m in eds.maps.items()}
    return max(scores, key=scores.get)


# ------------------------------------------------------------- quantification
def denoise_counts(eds: EDSStack, sigma_nm: float = 30.0) -> EDSStack:
    """3D Gaussian smoothing of count maps with a physical kernel."""
    sig = [sigma_nm / v for v in eds.voxel_size.as_tuple()]
    return EDSStack({k: ndi.gaussian_filter(v, sig) for k, v in eds.maps.items()},
                    eds.voxel_size, eds.z_index)


def composition(eds: EDSStack, k_factors: Optional[dict] = None,
                exclude: Sequence[str] = ()) -> EDSStack:
    """Relative composition (fractions summing to 1) per voxel, with optional
    Cliff–Lorimer-style k-factors to convert intensity ratios to at.%."""
    els = [e for e in eds.elements if e not in exclude]
    k = k_factors or {}
    w = np.stack([eds.maps[e] * k.get(e, 1.0) for e in els])
    tot = w.sum(0, keepdims=True) + 1e-6
    return EDSStack({e: w[i] / tot[0] for i, e in enumerate(els)}, eds.voxel_size, eds.z_index)


def chemical_phases(eds: EDSStack, n_phases: int = 3, method: str = "kmeans",
                    sample: int = 200_000, mask: Optional[np.ndarray] = None
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    """Unsupervised chemical phase map from composition vectors.

    Returns labels (1..n, 0 = outside mask) and a table of mean composition
    per phase, phases sorted by their dominant element.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF
    from sklearn.mixture import GaussianMixture

    X = eds.stacked().reshape(-1, len(eds.elements))
    valid = np.ones(len(X), bool) if mask is None else mask.reshape(-1)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(valid)
    tr = X[rng.choice(idx, min(sample, len(idx)), replace=False)]
    if method == "kmeans":
        model = KMeans(n_phases, n_init=10, random_state=0).fit(tr)
        pred = model.predict(X)
    elif method == "gmm":
        model = GaussianMixture(n_phases, covariance_type="full", random_state=0).fit(tr)
        pred = model.predict(X)
    elif method == "nmf":
        model = NMF(n_phases, init="nndsvda", max_iter=500, random_state=0).fit(np.clip(tr, 0, None))
        pred = np.argmax(model.transform(np.clip(X, 0, None)), 1)
    else:
        raise ValueError(method)
    lab = np.where(valid, pred + 1, 0).reshape(eds.shape).astype(np.uint8)
    rows = []
    for p in range(1, n_phases + 1):
        sel = (lab == p)
        r = {"phase": p, "volume_fraction": float(sel.sum() / max(valid.sum(), 1))}
        for i, e in enumerate(eds.elements):
            r[e] = float(eds.maps[e][sel].mean()) if sel.any() else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df["dominant"] = df[eds.elements].idxmax(axis=1)
    return lab, df


def composition_by_label(eds: EDSStack, labels: np.ndarray, names: Optional[dict] = None,
                         z_index: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Mean ± std of each element inside each label (SE class or object).
    ``labels`` is on the SE grid; only SE slices that carry EDS are used."""
    z_index = eds.z_index if z_index is None else z_index
    lab = labels[np.asarray(z_index)]
    lab = lab[:, : eds.shape[1], : eds.shape[2]]
    ids = [int(i) for i in np.unique(lab) if i != 0]
    rows = []
    for i in ids:
        sel = lab == i
        r = {"label": i, "name": (names or {}).get(str(i), str(i)), "voxels": int(sel.sum())}
        for e in eds.elements:
            v = eds.maps[e][sel]
            r[f"{e}_mean"], r[f"{e}_std"] = float(v.mean()), float(v.std())
        rows.append(r)
    return pd.DataFrame(rows)


def element_rgb(eds: EDSStack, elements: Sequence[str], z: int,
                pct: float = 99.5) -> np.ndarray:
    """RGB composite (up to 3 elements → R, G, B) for one EDS slice."""
    rgb = np.zeros(eds.shape[1:] + (3,), np.float32)
    for c, e in enumerate(elements[:3]):
        m = eds.maps[e][z]
        rgb[..., c] = np.clip(m / (np.percentile(eds.maps[e], pct) + 1e-6), 0, 1)
    return rgb


@op("chem.eds_tomography", inputs=("volume", "labels"))
def eds_tomography(volume: Volume, labels: Optional[Volume] = None, eds_root: str = "",
                   eds_voxel_nm: Optional[Sequence[float]] = None,
                   eds_every: int = 1, n_phases: int = 3, sigma_nm: float = 30.0,
                   method: str = "kmeans") -> dict:
    """Import EDS element maps, register to SE, cluster chemical phases, fuse with labels."""
    vs = VoxelSize(*eds_voxel_nm) if eds_voxel_nm else volume.voxel_size.scaled(eds_every, 1, 1)
    eds = read_eds_folders(eds_root, vs, z_index=None)
    eds.z_index = np.arange(eds.shape[0]) * eds_every
    reg = denoise_counts(register_to_se(eds, volume), sigma_nm)
    comp = composition(reg)
    lab, table = chemical_phases(comp, n_phases, method)
    out = {"phases_table": table,
           "chem_phases": Volume(lab, reg.voxel_size, "chem_phases", "labels",
                                 {"classes": {str(r.phase): r.dominant
                                              for r in table.itertuples()}}),
           "registration": pd.DataFrame(reg.shifts_px, columns=["dy_px", "dx_px"]),
           "eds": reg}
    if labels is not None:
        out["composition_by_class"] = composition_by_label(comp, labels.numpy(),
                                                           labels.meta.get("classes"))
    return out


__all__ = ["EDSStack", "read_eds_folders", "register_to_se", "denoise_counts", "composition",
           "chemical_phases", "composition_by_label", "element_rgb", "eds_tomography"]
