"""Compare many samples (not available in Avizo).

Each sample is summarised with the same metrics and, crucially, the same
uncertainty model: a phase fraction measured in a finite FIB volume has a
sampling error set by the integral range, so two samples are only
"different" if the gap exceeds what re-sampling the same material would give.
Particle-size distributions are compared with Mann–Whitney and
Kolmogorov–Smirnov tests and Cliff's delta (effect size). P-values are
Holm-corrected across the comparisons made.

    nuscope compare A=projects/a/recipe_run B=projects/b/recipe_run
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .analyze.confidence import integral_range_cov
from .analyze.phase import _area
from .core.volume import Volume


def summarize(labels: Volume, particles: Optional[pd.DataFrame] = None,
              phases: Optional[Sequence[int]] = None, max_voxels: int = 8_000_000) -> dict:
    """Metrics with sampling standard errors for one sample."""
    lab = labels.numpy()
    sp = labels.voxel_size.as_tuple()
    names = labels.meta.get("classes", {})
    phases = list(phases or [int(c) for c in np.unique(lab) if c != 0])
    pv = labels.preview(max_voxels, isotropic=True) if lab.size > max_voxels else labels
    small = pv.numpy()
    V = lab.size * np.prod(sp) / 1e9
    rows = []
    for c in phases:
        m = lab == c
        p = float(m.mean())
        A3 = integral_range_cov(small == c, pv.voxel_size.as_tuple())
        se = np.sqrt(p * (1 - p) * A3 / V) if np.isfinite(A3) else np.nan
        sv = _area(m, sp) / (lab.size * np.prod(sp)) * 1e3
        rows.append({"phase": c, "name": names.get(str(c), str(c)), "fraction": p,
                     "fraction_se": se, "sv_um-1": sv,
                     # S/V scales with the same sampling statistics to first order
                     "sv_se": sv * se / p if p else np.nan, "volume_um3": V})
    return {"phases": pd.DataFrame(rows), "particles": particles}


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) * len(b) > 4e6:  # subsample for speed
        rng = np.random.default_rng(0)
        a = rng.choice(a, min(len(a), 2000), replace=False)
        b = rng.choice(b, min(len(b), 2000), replace=False)
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)


def compare(samples: dict[str, dict], size_col: str = "eq_diameter_nm") -> dict:
    """Pairwise tests between summarised samples (``summarize`` outputs)."""
    rows = []
    for (na, a), (nb, b) in combinations(samples.items(), 2):
        pa, pb = a["phases"].set_index("phase"), b["phases"].set_index("phase")
        for c in pa.index.intersection(pb.index):
            for metric, se in (("fraction", "fraction_se"), ("sv_um-1", "sv_se")):
                d = pb.loc[c, metric] - pa.loc[c, metric]
                s = np.hypot(pa.loc[c, se], pb.loc[c, se])
                z = d / s if s > 0 else np.nan
                mdd = 2.8 * s  # minimum detectable difference: alpha 0.05, power 0.8
                # volume per sample needed for this difference to reach that detectability
                v_need = (pa.loc[c, "volume_um3"] * (mdd / abs(d)) ** 2) if d else np.inf
                rows.append({"a": na, "b": nb, "phase": pa.loc[c, "name"], "metric": metric,
                             "value_a": pa.loc[c, metric], "value_b": pb.loc[c, metric],
                             "difference": d, "rel_difference": d / pa.loc[c, metric],
                             "z": z, "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z)
                             else np.nan, "test": "sampling-error z",
                             "min_detectable_diff": mdd,
                             "min_detectable_rel": mdd / pa.loc[c, metric],
                             "volume_needed_um3": v_need})
        A, B = a.get("particles"), b.get("particles")
        if A is not None and B is not None and len(A) > 5 and len(B) > 5:
            xa, xb = A[size_col].dropna().to_numpy(), B[size_col].dropna().to_numpy()
            mw = stats.mannwhitneyu(xa, xb)
            ks = stats.ks_2samp(xa, xb)
            rows.append({"a": na, "b": nb, "phase": "particles", "metric": f"{size_col} (median)",
                         "value_a": float(np.median(xa)), "value_b": float(np.median(xb)),
                         "difference": float(np.median(xb) - np.median(xa)),
                         "rel_difference": float(np.median(xb) / np.median(xa) - 1),
                         "z": np.nan, "p": float(mw.pvalue), "test": "Mann-Whitney U",
                         "ks_p": float(ks.pvalue), "cliffs_delta": cliffs_delta(xb, xa)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = _holm(df["p"].fillna(1).to_numpy())
        df["significant"] = df["p_holm"] < 0.05
    summary = pd.concat([s["phases"].assign(sample=n) for n, s in samples.items()],
                        ignore_index=True)
    return {"tests": df, "summary": summary}


def load_project(path: str | Path, labels: str = "volumes/phases.zarr",
                 particles: str = "tables/particles_measured.csv") -> dict:
    path = Path(path)
    lab = Volume.from_zarr(path / labels)
    pt = pd.read_csv(path / particles) if (path / particles).exists() else None
    if pt is not None and "touches_border" in pt:
        pt = pt[~pt.touches_border]
    return summarize(lab, pt)


def plot(result: dict, out: Optional[str] = None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .viz import PALETTE
    from .viz.figures import STYLE
    s = result["summary"]
    samples = list(dict.fromkeys(s["sample"]))
    phases = list(dict.fromkeys(s["name"]))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(16, 0.6 * len(phases) * len(samples) + 2))
        for k, (metric, se, lab) in enumerate((("fraction", "fraction_se", "volume fraction"),
                                               ("sv_um-1", "sv_se", "S/V (µm⁻¹)"))):
            y = 0
            ticks, names = [], []
            for ph in phases:
                for i, smp in enumerate(samples):
                    r = s[(s["name"] == ph) & (s["sample"] == smp)]
                    if r.empty:
                        continue
                    v, e = float(r[metric].iloc[0]), 2 * float(r[se].iloc[0])
                    ax[k].errorbar(v, y, xerr=e, fmt="o", color=PALETTE[i % 10], capsize=4,
                                   label=smp if (ph == phases[0]) else None)
                    ticks.append(y)
                    names.append(f"{ph} · {smp}")
                    y += 1
                y += 0.8
            ax[k].set_yticks(ticks, names, fontsize=9)
            ax[k].invert_yaxis()
            ax[k].set_xlabel(f"{lab} (±95 % sampling)")
        ax[0].legend(frameon=False, loc="lower right")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
        fig.tight_layout()
        if out:
            fig.savefig(out, dpi=130, bbox_inches="tight")
        return fig


__all__ = ["summarize", "compare", "load_project", "plot", "cliffs_delta"]
