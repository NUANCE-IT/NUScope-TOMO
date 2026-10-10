"""Sparse / fused chemical imaging benchmark on sample 335 geometry (simulated EDS).

    python examples/chemical_fusion.py [project_dir]

At matched EDS acquisition time (~6-10 % of a full map) compares:
  * context-driven sparse EDS (deep kernel learning on real SE patches),
  * coupled-NMF pan-sharpening (Borodinov, Kalinin, Ovchinnikova et al.,
    npj Comput. Mater. 2019) of a coarse EDS map with BSE (physically linked,
    Castaing's rule), optionally plus a fast short-dwell high-resolution map,
  * the same fusion with the SE image (no physical link to chemistry),
  * a coarse map upsampled, and the full map (100 % time).

Two chemistries are placed on the real segmented geometry:
  A  NMC-like (Mn / Ni: almost no BSE contrast between the phases),
  B  SOFC-like (Ni / YSZ / LSM: strong BSE contrast).
Both contain chemistry the SE image cannot see: smooth composition
variations and a hidden region with the same SE contrast as its host phase.
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from nuscope_tomo import Volume  # noqa: E402
from nuscope_tomo.sparse import (active_sparse, bse_response, cnmf_pansharpen,  # noqa: E402
                                 fusion_metrics)
from nuscope_tomo.viz import PALETTE  # noqa: E402
from nuscope_tomo.viz.figures import STYLE  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
V, T, OUT = PROJ / "volumes", PROJ / "tables", PROJ / "gallery"
se_vol, ph_vol = Volume.from_zarr(V / "denoised_n2v.zarr"), Volume.from_zarr(V / "phases.zarr")
BE, BC, COUNTS, FAST = 6, 4, 40, 0.04  # fine EDS px = 6 SE px; coarse = 4x4 fine; dwell of fast map
SCENARIOS = {
    "A  NMC-like (weak BSE contrast)": dict(
        els=["C", "Co", "Mn", "Ni", "O"],
        comp={1: {"C": .7, "O": .2, "Ni": .05, "Mn": .05}, 2: {"Ni": .15, "Mn": .55, "Co": .05, "O": .25},
              3: {"Ni": .45, "Mn": .15, "Co": .15, "O": .25}, 4: {"Ni": .60, "Mn": .05, "Co": .10, "O": .25}},
        hidden={"Ni": .25, "Mn": .15, "Co": .35, "O": .25}, swap=("Ni", "Mn"), probe="Co", k=4),
    "B  SOFC-like (strong BSE contrast)": dict(
        els=["C", "La", "Mn", "Ni", "O", "Y", "Zr"],
        comp={1: {"C": .8, "O": .2}, 2: {"Zr": .28, "Y": .06, "O": .66}, 3: {"Ni": .95, "O": .05},
              4: {"La": .2, "Mn": .2, "O": .6}},
        hidden={"Ni": .40, "Zr": .55, "O": .05}, swap=("Zr", "Y"), probe="Zr", k=5),  # BSE-visible
}
METHODS = ["full map", "coarse 6 %, upsampled", "DKL sparse 10 % (SE context)",
           "DKL sparse 10 % (SE + BSE context)",
           "CNMF: coarse 6 % + BSE", "CNMF: coarse 6 % + BSE + fast 4 %",
           "CNMF: coarse 6 % + SE (no physical link)"]
TIME = dict(zip(METHODS, [1.0, 1 / BC ** 2, 0.10, 0.10, 1 / BC ** 2, 1 / BC ** 2 + FAST, 1 / BC ** 2]))


def to_grid(a, b):
    h, w = a.shape[0] // b * b, a.shape[1] // b * b
    return a[:h, :w].reshape(h // b, b, w // b, b, *a.shape[2:]).mean((1, 3))


def norm(k):
    return k / np.maximum(k.sum(-1, keepdims=True), 1)


def truth_chemistry(lab, rng, sc):
    els = sc["els"]
    comp = np.zeros(lab.shape + (len(els),), np.float32)
    for c, d in sc["comp"].items():
        comp[lab == c] = [d.get(e, 0) for e in els]
    mod = ndi.gaussian_filter(rng.standard_normal(lab.shape), 30)
    mod = 0.05 * mod / (np.abs(mod).max() + 1e-9)
    a, b = sc["swap"]
    comp[..., els.index(a)] += np.where(lab >= 2, mod, 0)
    comp[..., els.index(b)] -= np.where(lab >= 2, mod, 0)
    blob = ndi.gaussian_filter(rng.standard_normal(lab.shape), 25)
    hidden = (blob > np.percentile(blob, 88)) & (lab == 3)
    comp[hidden] = [sc["hidden"].get(e, 0) for e in els]
    comp = np.clip(comp, 0, None)
    return comp / comp.sum(-1, keepdims=True), hidden


def run_slice(sc, z, rows, keep):
    rng = np.random.default_rng(z)
    els = sc["els"]
    lab = ph_vol.numpy()[z]
    se = se_vol.get_slice(0, z).astype(np.float32)
    comp, hidden = truth_chemistry(lab, rng, sc)
    blurred = np.stack([ndi.gaussian_filter(comp[..., e], 6) for e in range(len(els))], -1)
    hF, wF = (lab.shape[0] // (BE * BC)) * BC, (lab.shape[1] // (BE * BC)) * BC
    truth = to_grid(blurred, BE)[:hF, :wF]
    hid_f = to_grid(hidden.astype(float), BE)[:hF, :wF] > 0.5
    se_c = se[: hF * BE, : wF * BE]
    blocks = lab[: hF * BE, : wF * BE].reshape(hF, BE, wF, BE).transpose(0, 2, 1, 3).reshape(hF, wF, -1)
    lab_f = np.apply_along_axis(lambda v: np.bincount(v.astype(int), minlength=5).argmax(), 2, blocks)
    full = norm(rng.poisson(truth * COUNTS))
    coarse = norm(rng.poisson(to_grid(truth, BC) * COUNTS))
    fast = rng.poisson(truth * COUNTS * FAST).astype(np.float64)
    R_bse = bse_response(els)
    # BSE at SE resolution (small probe blur), then on the EDS grid for the fusion
    bse_hi = np.stack([ndi.gaussian_filter(comp[..., e], 1.5) for e in range(len(els))],
                      -1)[: hF * BE, : wF * BE] @ R_bse[0]
    bse_hi = bse_hi + rng.normal(0, 0.02 * np.ptp(bse_hi), bse_hi.shape)
    bse = to_grid(bse_hi, BE)
    se_f = to_grid(se_c, BE)
    cent = np.array([[sc["comp"][c].get(e, 0) for e in els] for c in (1, 2, 3, 4)] +
                    [[sc["hidden"].get(e, 0) for e in els]])
    cent /= cent.sum(1, keepdims=True)
    lab_true = np.where(hid_f, 5, lab_f)
    p = els.index(sc["probe"])
    host = (lab_f == 3) & ~hid_f
    true_con = truth[..., p][hid_f].mean() - truth[..., p][host].mean()

    def score(name, est):
        m = fusion_metrics(est, truth)
        cls = np.argmin(((est[..., None, :] - cent) ** 2).sum(-1), -1) + 1
        rows.append({"scenario": SC_NAME, "slice": z, "method": name, "eds_time": TIME[name], **m,
                     "phase_acc": float((cls == lab_true).mean()),
                     "hidden_recall": float((cls[hid_f] == 5).mean()),
                     "hidden_contrast": float((est[..., p][hid_f].mean() - est[..., p][host].mean())
                                              / true_con)})
        if z == 400:
            keep[name] = est

    score("full map", full)
    score("coarse 6 %, upsampled", ndi.zoom(coarse, (BC, BC, 1), order=1)[:hF, :wF])
    flat = full.reshape(-1, len(els))
    r = active_sparse(se_c, (hF, wF), BE, lambda i: flat[i], budget=0.10, init=0.025, rounds=3,
                      seed=z)
    score("DKL sparse 10 % (SE context)", np.clip(r["mean"], 0, None))
    r2 = active_sparse(np.stack([se_c, bse_hi]), (hF, wF), BE, lambda i: flat[i], budget=0.10,
                       init=0.025, rounds=3, seed=z)
    score("DKL sparse 10 % (SE + BSE context)", np.clip(r2["mean"], 0, None))
    k = sc["k"]
    score("CNMF: coarse 6 % + BSE", cnmf_pansharpen(coarse, bse[..., None], R_bse, BC,
                                                     n_components=k)["cube"])
    YM = np.concatenate([bse[..., None], fast], -1)
    R = np.vstack([R_bse, COUNTS * FAST * np.eye(len(els))])
    score("CNMF: coarse 6 % + BSE + fast 4 %",
          cnmf_pansharpen(coarse, YM, R, BC, n_components=k,
                          channel_weights=np.r_[3.0, 0.3 * np.ones(len(els))])["cube"])
    r_se, *_ = np.linalg.lstsq(coarse.reshape(-1, len(els)), to_grid(se_f, BC).ravel(), rcond=None)
    score("CNMF: coarse 6 % + SE (no physical link)",
          cnmf_pansharpen(coarse, se_f[..., None], r_se[None, :], BC, n_components=k)["cube"])
    if z == 400:
        keep.update(truth=truth, hidden=hid_f, se=se_f, bse=bse, probe=p, els=els)


rows, keeps = [], {}
for SC_NAME, sc in SCENARIOS.items():
    keeps[SC_NAME] = {}
    for z in (200, 400, 600):
        t = time.time()
        run_slice(sc, z, rows, keeps[SC_NAME])
        print(SC_NAME, z, f"{time.time() - t:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(T / "chemical_fusion.csv", index=False)
agg = df.groupby(["scenario", "method"], sort=False)[["eds_time", "rmse", "cc", "sam", "phase_acc",
                                                      "hidden_recall", "hidden_contrast"]].mean()
agg.to_csv(T / "chemical_fusion_summary.csv")
print(agg.round(3).to_string())

short = {"full map": "full map 100 %", "coarse 6 %, upsampled": "coarse 6 % upsampled",
         "DKL sparse 10 % (SE context)": "DKL 10 % (SE)",
         "DKL sparse 10 % (SE + BSE context)": "DKL 10 % (SE+BSE)", "CNMF: coarse 6 % + BSE": "CNMF + BSE 6 %",
         "CNMF: coarse 6 % + BSE + fast 4 %": "CNMF + BSE + fast 10 %",
         "CNMF: coarse 6 % + SE (no physical link)": "CNMF + SE 6 %"}
with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(4, 8, height_ratios=[1, 1, 1.15, 1.15], hspace=0.35, wspace=0.06)
    for row, (sname, kp) in enumerate(keeps.items()):
        p, els = kp["probe"], kp["els"]
        tiles = [("SE (real)", kp["se"], "gray"), ("BSE (sim.)", kp["bse"], "gray"),
                 (f"{els[p]} truth", kp["truth"][..., p], "viridis")] + \
                [(f"{els[p]} · {short[m]}", kp[m][..., p], "viridis")
                 for m in ("full map", "coarse 6 %, upsampled", "DKL sparse 10 % (SE context)",
                           "DKL sparse 10 % (SE + BSE context)", "CNMF: coarse 6 % + BSE")]
        vmax = np.percentile(kp["truth"][..., p], 99.5)
        for i, (t_, im, cm) in enumerate(tiles):
            a = fig.add_subplot(gs[row, i])
            a.imshow(im, cmap=cm, vmin=0 if cm != "gray" else None,
                     vmax=vmax if cm != "gray" else None, interpolation="nearest")
            if i == 2:
                a.contour(kp["hidden"], [0.5], colors="white", linewidths=0.7)
            a.set_title(t_, loc="left", fontsize=9)
            a.set_xticks([]), a.set_yticks([])
        a0 = fig.axes[-8]
        a0.text(-0.08, 0.5, sname.split("(")[0], transform=a0.transAxes, rotation=90, va="center",
                ha="right", fontsize=11, fontweight="bold", color="white")
    for j, (col, lab_) in enumerate((("rmse", "composition RMSE ↓"), ("phase_acc", "phase accuracy ↑"),
                                      ("hidden_contrast", "hidden region recovered (1 = perfect) ↑"),
                                      ("sam", "spectral angle (rad) ↓"))):
        a = fig.add_subplot(gs[2:, 2 * j:2 * j + 2])
        y = np.arange(len(METHODS))
        for s_i, sname in enumerate(SCENARIOS):
            vals = agg.loc[sname].loc[METHODS, col].to_numpy()
            a.barh(y + (s_i - 0.5) * 0.38, vals, 0.36, color=PALETTE[[1, 0][s_i]],
                   label=sname.split("(")[0].strip())
        a.set_yticks(y, [short[m] for m in METHODS] if j == 0 else [""] * len(METHODS), fontsize=9)
        a.invert_yaxis()
        a.set_title(lab_, loc="left", fontsize=10)
        if j == 0:
            a.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Chemical imaging at a fraction of full-map EDS time: sparse DKL vs coupled-NMF "
                 "pan-sharpening (Borodinov et al. 2019). SIMULATED EDS with SE-invisible chemistry "
                 "on the real sample 335 geometry", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "24_chemical_fusion.png", dpi=110, bbox_inches="tight")
