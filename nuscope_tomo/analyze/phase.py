"""Phase-level microstructure metrics (materials-science quantities that
Avizo exposes through separate modules): volume fractions, specific surface
area, interfacial areas, connectivity/percolation, geometric tortuosity and
continuous size distributions (local thickness)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import euler_number, marching_cubes, mesh_surface_area

from ..core.registry import op
from ..core.volume import Volume


def _area(mask: np.ndarray, spacing) -> float:
    if not mask.any() or mask.all():
        return 0.0
    # no padding: the cut faces of the field of view are not interfaces
    v, f, _, _ = marching_cubes(ndi.gaussian_filter(mask.astype(np.float32), 1.0), 0.5,
                                spacing=spacing)
    return float(mesh_surface_area(v, f))


def percolation(mask: np.ndarray) -> dict:
    """Fraction of the phase in the largest cluster and whether a single
    cluster connects opposite faces along z, y and x."""
    lab, n = ndi.label(mask)
    if n == 0:
        return {"n_clusters": 0, "largest_cluster_frac": 0.0,
                "percolates_z": False, "percolates_y": False, "percolates_x": False}
    sizes = np.bincount(lab.ravel())[1:]
    out = {"n_clusters": int(n), "largest_cluster_frac": float(sizes.max() / sizes.sum())}
    for ax, name in enumerate("zyx"):
        a = set(np.unique(np.take(lab, 0, axis=ax))) - {0}
        b = set(np.unique(np.take(lab, -1, axis=ax))) - {0}
        out[f"percolates_{name}"] = bool(a & b)
    return out


def geometric_tortuosity(mask: np.ndarray, spacing, axis: int = 0) -> float:
    """Mean shortest in-phase path length between the two faces normal to
    ``axis`` divided by the straight distance (geodesic, fast marching on
    the voxel graph). NaN if the phase does not percolate."""
    from skimage.graph import MCP_Geometric

    cost = np.where(mask, 1.0, np.inf)
    starts = np.argwhere(_face(mask, axis, 0))
    if len(starts) == 0:
        return float("nan")
    m = MCP_Geometric(cost, sampling=tuple(spacing))
    cum, _ = m.find_costs([tuple(p) for p in starts])
    end = np.take(cum, -1, axis=axis)
    endmask = np.take(mask, -1, axis=axis)
    vals = end[endmask & np.isfinite(end)]
    if vals.size == 0:
        return float("nan")
    straight = (mask.shape[axis] - 1) * spacing[axis]
    return float(np.mean(vals) / straight)


def _face(mask, axis, idx):
    sel = np.zeros_like(mask)
    sl = [slice(None)] * 3
    sl[axis] = idx
    sel[tuple(sl)] = mask[tuple(sl)]
    return sel


def size_distribution(mask: np.ndarray, spacing, radii_nm: Optional[Sequence[float]] = None
                      ) -> pd.DataFrame:
    """Continuous size distribution (Münch & Holzer 2008): fraction of the
    phase that can be covered by spheres of radius ≥ r (morphological
    opening via Euclidean distance transforms)."""
    spacing = np.asarray(spacing, float)
    dist = ndi.distance_transform_edt(mask, sampling=spacing)
    rmax = float(dist.max())
    if radii_nm is None:
        rmin = float(spacing.max())
        radii_nm = np.geomspace(rmin, max(rmax, rmin * 1.01), 16)
    total = mask.sum()
    rows = []
    for r in radii_nm:
        core = dist >= r
        if not core.any():
            rows.append((r, 0.0))
            continue
        opened = ndi.distance_transform_edt(~core, sampling=spacing) <= r
        rows.append((r, float((opened & mask).sum() / total)))
    df = pd.DataFrame(rows, columns=["radius_nm", "cumulative_fraction"])
    df["diameter_nm"] = 2 * df["radius_nm"]
    df["frequency"] = -np.gradient(df["cumulative_fraction"], np.log(df["radius_nm"]))
    return df


@op("analyze.phases", inputs=("labels",))
def phase_metrics(labels: Volume, tortuosity: bool = True, max_voxels_tortuosity: int = 96**3,
                  max_voxels: int = 400_000_000) -> dict:
    """Volume fraction, specific surface, interfaces, percolation, tortuosity, size distribution per class."""
    if np.prod(labels.shape) > max_voxels:  # surfaces on a strided copy for huge volumes
        labels = labels.preview(max_voxels)
    lab = labels.numpy()
    sp = labels.voxel_size.as_tuple()
    total_vol = lab.size * labels.voxel_size.voxel_volume_nm3
    names = labels.meta.get("classes", {})
    classes = [int(c) for c in np.unique(lab) if c != 0] or [0]
    rows, sizes, areas = [], [], {}
    # near-isotropic downsampled copy for the geodesic / size computations
    lv = labels if labels.kind == "labels" else labels.derive(lab, kind="labels")
    pv = lv.preview(max_voxels_tortuosity, isotropic=True)
    small, sp_small = pv.numpy(), pv.voxel_size.as_tuple()
    for c in classes:
        m = lab == c
        area = areas[c] = _area(m, sp)
        r = {"class": c, "name": names.get(str(c), f"class_{c}"),
             "volume_fraction": float(m.mean()),
             "volume_um3": float(m.sum() * labels.voxel_size.voxel_volume_nm3 / 1e9),
             "surface_area_um2": area / 1e6,
             "specific_surface_um-1": area / total_vol * 1e3,
             "euler_number": int(euler_number(small == c, connectivity=1))}
        r.update(percolation(m))
        if tortuosity:
            for ax, name in enumerate("zyx"):
                r[f"tortuosity_{name}"] = geometric_tortuosity(small == c, sp_small, ax)
        rows.append(r)
        sd = size_distribution(small == c, sp_small)
        sd.insert(0, "class", c)
        sd.insert(1, "name", r["name"])
        sizes.append(sd)
    inter = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            # interface between a and b: area of a∪… approximated from pair masks
            ab = max(0.0, areas[a] + areas[b] - _area((lab == a) | (lab == b), sp))
            inter.append({"class_a": a, "class_b": b,
                          "interface_area_um2": ab / 2 / 1e6,
                          "interface_density_um-1": ab / 2 / total_vol * 1e3})
    return {"phases": pd.DataFrame(rows), "size_distribution": pd.concat(sizes),
            "interfaces": pd.DataFrame(inter)}


__all__ = ["phase_metrics", "percolation", "geometric_tortuosity", "size_distribution"]
