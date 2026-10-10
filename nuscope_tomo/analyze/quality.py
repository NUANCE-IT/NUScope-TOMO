"""Data-health inspector for slice-and-view stacks (not available in Avizo).

Finds acquisition problems before they become analysis errors:

* per-slice brightness, contrast, noise (SNR), sharpness, curtaining index;
* blank, duplicated and outlier slices (robust z-scores);
* calibration check: compares the feature correlation length along the
  milling axis with the in-plane one. A large ratio usually means the slice
  thickness in the metadata is wrong (e.g. 50 nm recorded for 20 nm slices).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.restoration import estimate_sigma

from ..core.registry import op
from ..core.volume import Volume


def _slice_metrics(img: np.ndarray, prev: Optional[np.ndarray]) -> dict:
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, (2, 98))
    noise = float(estimate_sigma(a)) + 1e-6
    sm = ndi.gaussian_filter(a, 1.0)
    hp = a - ndi.gaussian_filter(a, 8)
    col = hp.mean(0)  # curtains run along Y (image columns)
    row = hp.mean(1)
    return {
        "mean": float(a.mean()), "p2": float(lo), "p98": float(hi),
        "snr": float((hi - lo) / noise),
        "sharpness": float(ndi.laplace(sm).var() / (sm.var() + 1e-6)),
        "curtaining": float(col.std() / (row.std() + 1e-6)),
        "diff_prev": float(np.abs(a - prev).mean()) if prev is not None else np.nan,
    }


def correlation_lengths(vol: np.ndarray, spacing_nm) -> np.ndarray:
    """1/e decay length (nm) of the intensity autocorrelation along z, y, x."""
    a = vol.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    F = np.fft.rfftn(a, s=[2 * n for n in a.shape])
    acf = np.fft.irfftn(np.abs(F) ** 2)[: a.shape[0], : a.shape[1], : a.shape[2]]
    acf /= acf[0, 0, 0]
    out = []
    for ax in range(3):
        prof = _axis_profile(acf, ax)
        below = np.flatnonzero(prof < np.exp(-1))
        if below.size == 0:
            out.append(np.nan)
            continue
        k = below[0]  # linear interpolation between k-1 and k
        f = (prof[k - 1] - np.exp(-1)) / (prof[k - 1] - prof[k])
        out.append((k - 1 + f) * spacing_nm[ax])
    return np.array(out)


def _axis_profile(acf: np.ndarray, ax: int) -> np.ndarray:
    idx = [0, 0, 0]
    idx[ax] = slice(None)
    return acf[tuple(idx)]


def calibration_check(volume: Volume, max_voxels: int = 160**3, ratio_warn: float = 1.8) -> dict:
    """Compare correlation length along the milling axis (Z) with XY."""
    vs = np.array(volume.voxel_size.as_tuple())
    nz, ny, nx = volume.shape
    # central block, roughly cubic in physical size, within the voxel budget
    side_nm = min(nz * vs[0], ny * vs[1], nx * vs[2])
    n = np.minimum(np.round(side_nm / vs).astype(int), (nz, ny, nx))
    while np.prod(n) > max_voxels:
        n = np.maximum(8, (n * 0.8).astype(int))
    c = np.array([nz, ny, nx]) // 2
    sl = tuple(slice(ci - ni // 2, ci - ni // 2 + ni) for ci, ni in zip(c, n))
    block = np.asarray(volume.data[sl])
    L = correlation_lengths(block, vs)
    ratio = L[0] / np.nanmean(L[1:])
    res = {"corr_len_z_nm": L[0], "corr_len_y_nm": L[1], "corr_len_x_nm": L[2],
           "z_over_xy": ratio, "slice_nm": vs[0], "warning": ""}
    if ratio > ratio_warn:
        res["warning"] = (f"Features are {ratio:.1f}x longer along the milling axis than in-plane. "
                          f"If the structure is not truly columnar, the slice thickness is likely "
                          f"~{vs[0] / ratio:.0f} nm rather than {vs[0]:.0f} nm.")
    elif ratio < 1 / ratio_warn:
        res["warning"] = (f"Features are {1 / ratio:.1f}x shorter along the milling axis; "
                          f"the slice thickness may be ~{vs[0] / ratio:.0f} nm.")
    return res


@op("qc.inspect")
def inspect(volume: Volume, step_xy: int = 2, z_score: float = 5.0) -> dict:
    """Per-slice quality metrics, flagged slices and a calibration check."""
    rows, prev = [], None
    for z in range(volume.shape[0]):
        img = volume.get_slice(0, z)[::step_xy, ::step_xy]
        r = _slice_metrics(img, prev)
        r["z"] = z
        rows.append(r)
        prev = img.astype(np.float32)
    df = pd.DataFrame(rows)
    flags = []
    for col in ["mean", "snr", "sharpness", "curtaining", "diff_prev"]:
        x = df[col].to_numpy()
        trend = ndi.median_filter(np.nan_to_num(x, nan=np.nanmedian(x)), 15, mode="nearest")
        r = x - trend
        mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))) + 1e-9
        z = r / mad
        df[f"{col}_z"] = z
        for i in np.flatnonzero(np.abs(np.nan_to_num(z)) > z_score):
            flags.append({"z": int(i), "metric": col, "value": float(x[i]),
                          "robust_z": float(z[i])})
    blank = df.index[df["p98"] - df["p2"] < 2].tolist()
    dup = df.index[df["diff_prev"] < 0.05 * np.nanmedian(df["diff_prev"])].tolist()
    for i in blank:
        flags.append({"z": int(i), "metric": "blank", "value": 0.0, "robust_z": np.nan})
    for i in dup:
        flags.append({"z": int(i), "metric": "duplicate", "value": float(df.diff_prev[i]),
                      "robust_z": np.nan})
    cal = calibration_check(volume)
    summary = {
        "slices": len(df), "flagged_slices": len({f["z"] for f in flags}),
        "snr_median": float(df.snr.median()), "sharpness_cv": float(df.sharpness.std() /
                                                                   df.sharpness.mean()),
        "brightness_drift_pct": float(100 * (df["mean"].iloc[-20:].mean() -
                                            df["mean"].iloc[:20].mean()) / df["mean"].mean()),
        "curtaining_median": float(df.curtaining.median()), **cal}
    return {"slices": df, "flags": pd.DataFrame(flags, columns=["z", "metric", "value",
                                                                "robust_z"]),
            "summary": pd.DataFrame([summary])}


__all__ = ["inspect", "calibration_check", "correlation_lengths"]
