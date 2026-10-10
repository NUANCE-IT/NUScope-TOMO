"""How much can the numbers be trusted? (not available in Avizo)

* ``rve`` — representative volume element via the integral-range method
  (Kanit et al., Int. J. Solids Struct. 40, 3647, 2003): the variance of a
  phase fraction measured in sub-volumes V decays as p(1-p)·A3/V; fitting
  the integral range A3 gives the RVE size for a target precision and the
  sampling error of the whole acquired volume.
* ``uncertainty`` — confidence intervals for volume fraction, specific
  surface and tortuosity factor from (a) sampling: integral range or block
  bootstrap, and (b) segmentation: moving every phase boundary by ±1 voxel.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ..core.registry import op
from ..core.volume import Volume
from .materials import effective_transport


def _box_sums(mask: np.ndarray) -> np.ndarray:
    """Summed-volume table with a zero border, for O(1) box sums."""
    s = np.zeros(tuple(n + 1 for n in mask.shape), np.int64)
    s[1:, 1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return s


def _box(s, z0, y0, x0, nz, ny, nx):
    z1, y1, x1 = z0 + nz, y0 + ny, x0 + nx
    return (s[z1, y1, x1] - s[z0, y1, x1] - s[z1, y0, x1] - s[z1, y1, x0]
            + s[z0, y0, x1] + s[z0, y1, x0] + s[z1, y0, x0] - s[z0, y0, x0])


def fraction_variance(mask: np.ndarray, spacing_nm, n_sizes: int = 10, n_boxes: int = 200,
                      seed: int = 0) -> pd.DataFrame:
    """Mean and variance of the phase fraction in random cubes of growing size."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing_nm, float)
    shape = np.array(mask.shape)
    max_edge = (shape * sp).min()
    S = _box_sums(mask)
    rows = []
    for L in np.geomspace(max(sp.max() * 4, max_edge / 40), max_edge * 0.95, n_sizes):
        n = np.maximum(1, np.round(L / sp).astype(int))
        n = np.minimum(n, shape)
        z0 = rng.integers(0, shape[0] - n[0] + 1, n_boxes)
        y0 = rng.integers(0, shape[1] - n[1] + 1, n_boxes)
        x0 = rng.integers(0, shape[2] - n[2] + 1, n_boxes)
        f = _box(S, z0, y0, x0, *n) / np.prod(n)
        rows.append({"edge_um": float((n * sp).prod() ** (1 / 3) / 1e3),
                     "volume_um3": float(np.prod(n * sp) / 1e9),
                     "mean": float(f.mean()), "var": float(f.var(ddof=1))})
    return pd.DataFrame(rows)


def integral_range(df: pd.DataFrame, p: float, v_total_um3: Optional[float] = None,
                   max_frac: float = 0.125) -> float:
    """Integral range A3 (µm³) from sub-volume variances.

    Sub-volumes drawn from one finite sample scatter around the *sample*
    mean, so their variance is D²(V) = p(1-p)·A3·(1/V − 1/V_total). Only
    sub-volumes up to ``max_frac`` of the total are used.
    """
    vt = v_total_um3 or df["volume_um3"].max() * 1.0001
    d = df[(df["volume_um3"] <= max_frac * vt) & (df["var"] > 0)]
    d = d.iloc[len(d) // 2:]  # large enough for the asymptotic 1/V regime
    if d.empty or p <= 0 or p >= 1:
        return np.nan
    return float(np.median(d["var"] / (p * (1 - p) * (1 / d["volume_um3"] - 1 / vt))))


def integral_range_cov(mask: np.ndarray, spacing_nm) -> float:
    """Integral range A3 (µm³) = ∫C(r)dr / p(1-p), integrating the unbiased
    indicator auto-covariance out to the first zero of its radial average.

    More reliable than sub-volume variances for FIB-SEM volumes, which are
    rarely large enough for the 1/V asymptote (validated on phantoms
    against the scatter of independent realisations: within ~20 %).
    """
    sp = np.asarray(spacing_nm, float)
    m = mask.astype(np.float32)
    p = float(m.mean())
    if p <= 0 or p >= 1:
        return np.nan
    a = m - p
    s = [2 * n for n in m.shape]
    C = np.fft.irfftn(np.abs(np.fft.rfftn(a, s=s)) ** 2, s=s)
    N = np.fft.irfftn(np.abs(np.fft.rfftn(np.ones_like(a), s=s)) ** 2, s=s)
    C = np.fft.fftshift(C / np.maximum(N, 1))
    c0 = np.array([n // 2 for n in C.shape])
    g = np.indices(C.shape, sparse=True)
    r = np.sqrt(sum(((g[i] - c0[i]) * sp[i]) ** 2 for i in range(3)))
    rb = (r / sp.max()).astype(np.int64)
    prof = ndi.mean(C, rb, np.arange(rb.max() + 1))
    k = int(np.argmax(prof <= 0)) if (prof <= 0).any() else len(prof)
    return float(C[rb < k].sum() * np.prod(sp) / 1e9 / (p * (1 - p)))


@op("analyze.rve", inputs=("labels",))
def rve(labels: Volume, phases: Optional[Sequence[int]] = None, rel_error: float = 0.05,
        max_voxels: int = 8_000_000) -> dict:
    """Representative volume and whole-volume sampling error of each phase fraction."""
    lv = labels.preview(max_voxels) if np.prod(labels.shape) > max_voxels else labels
    lab = lv.numpy()
    sp = lv.voxel_size.as_tuple()
    names = labels.meta.get("classes", {})
    phases = phases or [int(c) for c in np.unique(lab) if c != 0]
    V = lab.size * np.prod(sp) / 1e9
    rows, curves = [], []
    for c in phases:
        m = lab == c
        p = float(m.mean())
        dfv = fraction_variance(m, sp)
        A3 = integral_range_cov(m, sp)
        sd_full = np.sqrt(p * (1 - p) * A3 / V) if np.isfinite(A3) else np.nan
        v_rve = 4 * (1 - p) * A3 / (rel_error ** 2 * p) if np.isfinite(A3) else np.nan
        rows.append({"phase": c, "name": names.get(str(c), str(c)), "fraction": p,
                     "integral_range_um3": A3,
                     "ci95_low": p - 2 * sd_full, "ci95_high": p + 2 * sd_full,
                     "rel_error_full_volume": 2 * sd_full / p if p else np.nan,
                     f"rve_edge_um_{int(rel_error * 100)}pct": v_rve ** (1 / 3),
                     "volume_um3": V, "n_rve_in_volume": V / v_rve if v_rve else np.nan})
        dfv.insert(0, "phase", c)
        dfv["name"] = names.get(str(c), str(c))
        curves.append(dfv)
    return {"table": pd.DataFrame(rows), "curves": pd.concat(curves, ignore_index=True)}


# ------------------------------------------------------------------ uncertainty
def _shift_boundary(lab: np.ndarray, c: int, grow: bool) -> np.ndarray:
    """Move the boundary of phase c by one voxel (in-plane and through-plane)."""
    m = lab == c
    st = ndi.generate_binary_structure(3, 1)
    if grow:
        return ndi.binary_dilation(m, st) & (lab != 0)
    return ndi.binary_erosion(m, st, border_value=1)  # don't erode the volume faces


def _surface_per_volume(mask: np.ndarray, sp) -> float:
    from .phase import _area
    return _area(mask, sp) / (mask.size * np.prod(sp)) * 1e3  # µm⁻¹


@op("analyze.uncertainty", inputs=("labels",))
def uncertainty(labels: Volume, phases: Optional[Sequence[int]] = None, axis: int = 0,
                n_blocks: int = 6, max_voxels_transport: int = 8_000_000) -> pd.DataFrame:
    """Value ± sampling and segmentation uncertainty for ε, S/V and tortuosity factor.

    Volume fraction and S/V are evaluated at full resolution; the transport
    solve on a near-isotropic grid of ``max_voxels_transport``. Segmentation
    uncertainty moves every boundary by one *acquired* voxel.
    """
    lab = labels.numpy()
    sp = labels.voxel_size.as_tuple()
    names = labels.meta.get("classes", {})
    phases = phases or [int(c) for c in np.unique(lab) if c != 0]

    def coarse(mask):
        v = labels.derive(mask.astype(np.uint8), kind="labels")
        if np.prod(v.shape) > max_voxels_transport:
            v = v.preview(max_voxels_transport, isotropic=True)
        return v.numpy() > 0, v.voxel_size.as_tuple()

    ext = np.array(lab.shape) * sp
    bax = int([a for a in np.argsort(-ext) if a != axis][0])
    edges = np.linspace(0, lab.shape[bax], n_blocks + 1).astype(int)
    V_um3 = lab.size * np.prod(sp) / 1e9

    def blocks_of(mask):
        for b0, b1 in zip(edges[:-1], edges[1:]):
            sl = [slice(None)] * 3
            sl[bax] = slice(b0, b1)
            yield mask[tuple(sl)]

    def tau(mask):
        mc, spc = coarse(mask)
        return effective_transport(mc, spc, axis)["tortuosity_factor"]

    rows = []
    for c in phases:
        m = lab == c
        variants = {"grown": _shift_boundary(lab, c, True), "shrunk": _shift_boundary(lab, c, False)}
        metrics = {"volume_fraction": lambda mm: float(mm.mean()),
                   "specific_surface_um-1": lambda mm: _surface_per_volume(mm, sp),
                   f"tortuosity_{'zyx'[axis]}": tau}
        for name, fn in metrics.items():
            val = fn(m)
            if name == "volume_fraction":
                se = np.sqrt(val * (1 - val) * integral_range_cov(*coarse(m)) / V_um3)
                nb = 0
            else:
                bl = np.array([fn(b) for b in blocks_of(m)], float)
                bl = bl[np.isfinite(bl)]
                nb = len(bl)
                se = bl.std(ddof=1) / np.sqrt(nb) if nb > 1 else np.nan
            lo_b, hi_b = sorted([fn(variants["shrunk"]), fn(variants["grown"])])
            rows.append({"phase": c, "name": names.get(str(c), str(c)), "metric": name,
                         "value": val, "sampling_ci95_low": val - 2 * se,
                         "sampling_ci95_high": val + 2 * se,
                         "boundary_low": lo_b, "boundary_high": hi_b, "blocks": nb})
    return pd.DataFrame(rows)


__all__ = ["rve", "uncertainty", "fraction_variance", "integral_range", "integral_range_cov"]
