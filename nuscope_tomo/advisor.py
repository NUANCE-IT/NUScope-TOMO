"""Acquisition advisor: recommend pixel size, slice thickness, field of view
and milling depth from the first slices of a run (not available in Avizo).

Two complementary answers:

1. **Feature-scale rule.** Chord lengths of each phase give the size of the
   smallest features; a feature should span ≥ ``px_per_feature`` pixels in
   XY and ≥ ``slices_per_feature`` slices in Z to be measured reliably.
2. **Empirical coarsening test.** The data already acquired is re-sampled
   as if it had been taken with larger pixels (block majority) or thicker
   slices (every k-th plane). The coarsest sampling whose volume fractions,
   specific surface and mean chord stay within ``tolerance`` of the
   reference is the cheapest acquisition that would have given the same
   answer — with the imaging-time saving it implies.

Volume: the integral range of each phase gives the representative volume
for a target precision, turned into a required depth for the current field
of view (or a field of view for a fixed depth).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .analyze.confidence import integral_range_cov
from .analyze.phase import _area
from .core.registry import op
from .core.volume import Volume


# ------------------------------------------------------------------ chords
def chord_lengths(mask: np.ndarray, axis: int, spacing_nm: float) -> np.ndarray:
    """Lengths (nm) of runs of True along ``axis``, excluding runs cut by the edge."""
    m = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis]).astype(np.int8)
    pad = np.pad(m, ((0, 0), (1, 1)))
    d = np.diff(pad, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    L = ends[:, 1] - starts[:, 1]
    inner = (starts[:, 1] > 0) & (ends[:, 1] < m.shape[1])
    return L[inner] * spacing_nm


def _block_mode(lab: np.ndarray, f: int) -> np.ndarray:
    """Majority label in f×f blocks in XY (what a larger pixel would record)."""
    z, h, w = lab.shape
    h2, w2 = h // f * f, w // f * f
    lab = lab[:, :h2, :w2]
    classes = np.unique(lab)
    votes = np.stack([(lab == c).reshape(z, h2 // f, f, w2 // f, f).sum((2, 4))
                      for c in classes])
    return classes[np.argmax(votes, 0)].astype(lab.dtype)


def _metrics(lab: np.ndarray, sp, phases) -> dict:
    out = {}
    vol = lab.size * np.prod(sp)
    for c in phases:
        m = lab == c
        out[f"fraction_{c}"] = float(m.mean())
        out[f"sv_{c}"] = _area(m, sp) / vol * 1e3
        ch = np.concatenate([chord_lengths(m, ax, sp[ax]) for ax in (1, 2)])
        out[f"chord_{c}"] = float(ch.mean()) if ch.size else np.nan
    return out


# ------------------------------------------------------------------ advisor
@op("advise.acquisition", inputs=("labels",))
def advise(labels: Volume, phases: Optional[Sequence[int]] = None,
           tolerance: float = 0.05, target_rel_error: float = 0.05,
           px_per_feature: float = 4.0, slices_per_feature: float = 3.0,
           xy_factors: Sequence[int] = (1, 2, 3, 4), z_factors: Sequence[int] = (1, 2, 3, 4),
           max_voxels: int = 60_000_000) -> dict:
    """Recommend pixel size, slice thickness and volume from (early) segmented slices."""
    lab = labels.numpy()
    sp = labels.voxel_size.as_tuple()
    full_shape = lab.shape
    if lab.size > max_voxels:  # crop the field of view centrally: keep resolution on every axis
        f = np.sqrt(max_voxels / lab.size)
        h, w = int(lab.shape[1] * f), int(lab.shape[2] * f)
        y0, x0 = (lab.shape[1] - h) // 2, (lab.shape[2] - w) // 2
        lab_s = lab[:, y0:y0 + h, x0:x0 + w]
    else:
        lab_s = lab
    names = labels.meta.get("classes", {})
    phases = list(phases or [int(c) for c in np.unique(lab) if c != 0 and (lab == c).mean() > 0.01])

    # 1. feature-scale rule
    rows = []
    for c in phases:
        m = lab == c
        cxy = np.concatenate([chord_lengths(m, ax, sp[ax]) for ax in (1, 2)])
        cz = chord_lengths(m, 0, sp[0]) if lab.shape[0] > 4 else np.array([])
        cz_mean = float(cz.mean()) if cz.size else np.nan
        d10 = float(np.percentile(cxy, 10)) if cxy.size else np.nan
        rows.append({"phase": c, "name": names.get(str(c), str(c)),
                     "chord_xy_mean_nm": float(cxy.mean()) if cxy.size else np.nan,
                     "chord_xy_p10_nm": d10,
                     "chord_z_mean_nm": cz_mean,
                     "pixel_max_nm": d10 / px_per_feature,
                     "slice_max_nm": d10 / slices_per_feature})
    scale = pd.DataFrame(rows)

    # 2. empirical coarsening test
    ref = _metrics(lab_s, sp, phases)
    tests = []
    for fxy in xy_factors:
        lxy = _block_mode(lab_s, fxy) if fxy > 1 else lab_s
        for fz in z_factors:
            l2 = lxy[::fz]
            if l2.shape[0] < 3:
                continue
            sp2 = (sp[0] * fz, sp[1] * fxy, sp[2] * fxy)
            met = _metrics(l2, sp2, phases)
            dev = {k: abs(met[k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in ref}
            tests.append({"pixel_nm": sp2[1], "slice_nm": sp2[0], "xy_factor": fxy,
                          "z_factor": fz, "max_rel_change": max(dev.values()),
                          "worst_metric": max(dev, key=dev.get),
                          "imaging_time_factor": 1 / (fxy * fxy * fz),
                          "milling_time_factor": 1 / fz, **{f"d_{k}": v for k, v in dev.items()}})
    tests = pd.DataFrame(tests)
    ok = tests[tests.max_rel_change <= tolerance]
    best = ok.sort_values(["imaging_time_factor", "max_rel_change"]).iloc[0] if len(ok) else \
        tests.sort_values("max_rel_change").iloc[0]

    # 3. how much volume is needed
    V = lab.size * np.prod(sp) / 1e9
    area_um2 = lab.shape[1] * sp[1] * lab.shape[2] * sp[2] / 1e6
    vol_rows = []
    for c in phases:
        m = lab == c
        p = float(m.mean())
        A3 = integral_range_cov(m[:, ::2, ::2] if m.size > 2e7 else m,
                                (sp[0], sp[1] * (2 if m.size > 2e7 else 1),
                                 sp[2] * (2 if m.size > 2e7 else 1)))
        v_need = 4 * (1 - p) * A3 / (target_rel_error ** 2 * p) if np.isfinite(A3) else np.nan
        vol_rows.append({"phase": c, "name": names.get(str(c), str(c)), "fraction": p,
                         "integral_range_um3": A3, "volume_needed_um3": v_need,
                         "depth_needed_um_at_current_fov": v_need / area_um2,
                         "cube_edge_um": v_need ** (1 / 3),
                         "rel_error_now": 2 * np.sqrt(p * (1 - p) * A3 / V) / p})
    volume = pd.DataFrame(vol_rows)
    # an integral-range estimate needs a depth of several feature lengths
    depth_nm = lab.shape[0] * sp[0]
    # (z chords in a shallow volume are biased short, so use the in-plane feature length)
    reliable = bool(depth_nm >= 10 * np.nanmax(scale.chord_xy_mean_nm.fillna(0).to_numpy(),
                                               initial=0))

    rec = {
        "current_pixel_nm": sp[1], "current_slice_nm": labels.voxel_size.z,
        "recommended_pixel_nm": float(best.pixel_nm),
        "recommended_slice_nm": float(best.slice_nm),
        "imaging_speedup": float(1 / best.imaging_time_factor),
        "milling_speedup": float(1 / best.milling_time_factor),
        "max_change_at_recommendation": float(best.max_rel_change),
        "feature_rule_pixel_nm": float(scale.pixel_max_nm.min()),
        "feature_rule_slice_nm": float(scale.slice_max_nm.min()),
        "depth_needed_um": float(volume.depth_needed_um_at_current_fov.max()),
        "fov_area_um2": area_um2, "volume_now_um3": V,
        "target_rel_error": target_rel_error, "tolerance": tolerance,
        "depth_analysed_um": depth_nm / 1e3, "volume_estimate_reliable": reliable,
        "shape_analysed": "x".join(map(str, full_shape)),
    }
    rec["advice"] = _advice_text(rec, volume)
    return {"recommendation": pd.DataFrame([rec]), "feature_scale": scale,
            "coarsening": tests, "volume": volume}


def _advice_text(r: dict, volume: pd.DataFrame) -> str:
    parts = []
    if r["recommended_pixel_nm"] > r["current_pixel_nm"] or \
            r["recommended_slice_nm"] > r["current_slice_nm"]:
        parts.append(f"Sampling: {r['recommended_pixel_nm']:.0f} nm pixels and "
                     f"{r['recommended_slice_nm']:.0f} nm slices give the same fractions, "
                     f"surfaces and feature sizes within {100 * r['max_change_at_recommendation']:.1f} %, "
                     f"{r['imaging_speedup']:.0f}x less imaging and {r['milling_speedup']:.0f}x fewer slices.")
    else:
        parts.append("Sampling: the current pixel size and slice thickness are needed; coarser "
                     f"sampling changes results by more than {100 * r['tolerance']:.0f} %.")
    worst = volume.sort_values("depth_needed_um_at_current_fov").iloc[-1]
    pre = "" if r["volume_estimate_reliable"] else \
        f"(preliminary: only {r['depth_analysed_um']:.1f} µm milled so far) "
    parts.append(f"Volume {pre}: for ±{100 * r['target_rel_error']:.0f} % on every phase fraction, mill "
                 f"≈ {worst.depth_needed_um_at_current_fov:.0f} µm at this field of view "
                 f"({r['fov_area_um2']:.0f} µm²), or widen the field to a ≈ {worst.cube_edge_um:.0f} µm "
                 f"cube; limiting phase: {worst['name']}.")
    return " ".join(parts)


__all__ = ["advise", "chord_lengths"]
