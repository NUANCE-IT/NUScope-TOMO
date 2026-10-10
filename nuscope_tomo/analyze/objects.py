"""Per-object 3D morphometry in physical units (Avizo "Label Analysis").

All lengths are reported in nm (columns suffixed ``_nm``), volumes in nm³
and µm³, areas in nm². Anisotropic voxels are accounted for everywhere.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree
from skimage.measure import marching_cubes, mesh_surface_area

from ..core.registry import op
from ..core.volume import Volume

SMOOTH_SIGMA = 1.0

MEASURES = {
    "volume_um3": "Object volume (voxel count × voxel volume)",
    "surface_area_um2": "Marching-cubes surface area",
    "eq_diameter_nm": "Diameter of the sphere with the same volume",
    "sphericity": "π^(1/3)(6V)^(2/3)/A — 1 for a sphere",
    "solidity": "Volume / convex-hull volume",
    "feret_max_nm": "Maximum caliper diameter",
    "axis_major_nm / axis_mid_nm / axis_minor_nm": "Equivalent-ellipsoid axis lengths",
    "elongation": "axis_mid / axis_major",
    "flatness": "axis_minor / axis_mid",
    "theta_deg / phi_deg": "Major-axis polar angle from the milling (Z) direction, azimuth in XY",
    "centroid_*_um": "Centroid position",
    "nn_distance_nm": "Centroid nearest-neighbour distance",
    "touches_border": "True if truncated by the volume boundary",
    "mean_intensity / std_intensity": "Grey-value statistics inside the object",
    "<element>_mean": "Mean EDS signal inside the object (if element maps are given)",
}


def _object_row(lab_crop: np.ndarray, offset, spacing, label: int) -> dict:
    m = lab_crop == label
    sz, sy, sx = spacing
    n = int(m.sum())
    vol = n * sz * sy * sx
    coords = np.argwhere(m).astype(np.float64)
    phys = (coords + np.asarray(offset)) * spacing
    c = phys.mean(0)
    row = {"label": label, "voxels": n, "volume_um3": vol / 1e9,
           "centroid_z_um": c[0] / 1e3, "centroid_y_um": c[1] / 1e3,
           "centroid_x_um": c[2] / 1e3,
           "eq_diameter_nm": (6 * vol / np.pi) ** (1 / 3)}
    # surface (pad so closed surfaces are produced at crop borders)
    # light sub-voxel smoothing removes the ~8% staircase bias of binary marching cubes
    pm = ndi.gaussian_filter(np.pad(m, 2).astype(np.float32), SMOOTH_SIGMA)
    try:
        verts, faces, _, _ = marching_cubes(pm, 0.5, spacing=spacing)
        area = mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        verts, area = phys, np.nan
    row["surface_area_um2"] = area / 1e6
    row["sphericity"] = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area) if area else np.nan
    # inertia / equivalent ellipsoid
    if n >= 4:
        d = phys - c
        cov = d.T @ d / n + np.diag(np.square(spacing)) / 12.0
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals[::-1], 1e-12, None)
        evecs = evecs[:, ::-1]
        ax = 2 * np.sqrt(5 * evals)  # full axis lengths of a uniform ellipsoid
        row.update(axis_major_nm=ax[0], axis_mid_nm=ax[1], axis_minor_nm=ax[2],
                   elongation=ax[1] / ax[0], flatness=ax[2] / ax[1])
        v = evecs[:, 0] * np.sign(evecs[0, 0] or 1)
        row["theta_deg"] = float(np.degrees(np.arccos(np.clip(abs(v[0]), 0, 1))))
        row["phi_deg"] = float(np.degrees(np.arctan2(v[1], v[2])) % 180)
    # convex hull: solidity & Feret
    try:
        pts = verts if len(verts) >= 4 else phys
        hull = ConvexHull(pts)
        row["solidity"] = min(1.0, vol / hull.volume) if hull.volume > 0 else np.nan
        hp = pts[hull.vertices]
        if len(hp) > 1500:
            hp = hp[np.random.default_rng(0).choice(len(hp), 1500, replace=False)]
        dd = np.sqrt(((hp[:, None, :] - hp[None, :, :]) ** 2).sum(-1))
        row["feret_max_nm"] = float(dd.max())
    except Exception:
        row["solidity"] = np.nan
        row["feret_max_nm"] = np.nan
    return row


@op("analyze.objects", inputs=("labels", "intensity"))
def measure_objects(labels: Volume, intensity: Optional[Volume] = None,
                    elements: Optional[dict] = None, max_objects: int = 20000) -> pd.DataFrame:
    """Measure every labelled object (volume, area, shape, orientation, ...)."""
    lab = labels.numpy()
    spacing = np.array(labels.voxel_size.as_tuple(), float)
    slices = ndi.find_objects(lab)
    rows = []
    shape = np.array(lab.shape)
    for i, sl in enumerate(slices[:max_objects], 1):
        if sl is None:
            continue
        crop = lab[sl]
        r = _object_row(crop, [s.start for s in sl], spacing, i)
        r["touches_border"] = bool(any(s.start == 0 or s.stop == n for s, n in zip(sl, shape)))
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ids = df["label"].to_numpy()
    if intensity is not None:
        img = intensity.numpy().astype(np.float64)
        df["mean_intensity"] = ndi.mean(img, lab, ids)
        df["std_intensity"] = ndi.standard_deviation(img, lab, ids)
    for name, ev in (elements or {}).items():
        df[f"{name}_mean"] = ndi.mean(ev.numpy().astype(np.float64), lab, ids)
    if len(df) > 1:
        pts = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy() * 1e3
        d, _ = cKDTree(pts).query(pts, k=2)
        df["nn_distance_nm"] = d[:, 1]
    return df


def summarize(df: pd.DataFrame, exclude_border: bool = True) -> pd.DataFrame:
    """Population statistics (mean, std, quartiles) of each measure."""
    d = df[~df["touches_border"]] if exclude_border and "touches_border" in df else df
    num = d.select_dtypes("number").drop(columns=["label"], errors="ignore")
    out = num.describe(percentiles=[0.1, 0.5, 0.9]).T
    out.insert(0, "measure", out.index)
    return out.reset_index(drop=True)


__all__ = ["measure_objects", "summarize", "MEASURES"]
