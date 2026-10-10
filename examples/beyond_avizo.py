"""Analyses NuScope-TOMO adds beyond Avizo, run on sample 335.

    python examples/beyond_avizo.py [project_dir]

Needs the outputs of ``nuscope run examples/fib_sem_335.yaml``. Writes tables
to ``<project>/tables/beyond_*.csv`` and figures 16–19 to ``<project>/gallery``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nuscope_tomo import Volume  # noqa: E402
from nuscope_tomo.ai import denoise_n2v, unet_segment  # noqa: E402
from nuscope_tomo.analyze import inspect, rve, uncertainty  # noqa: E402
from nuscope_tomo.preprocess import auto_crop, bin_volume  # noqa: E402
from nuscope_tomo.segment import PixelClassifier, auto_annotations, smooth_labels  # noqa: E402
from nuscope_tomo.viz import PALETTE  # noqa: E402
from nuscope_tomo.viz.figures import STYLE, label_cmap  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
V, T, OUT = PROJ / "volumes", PROJ / "tables", PROJ / "gallery"
BANDS = [[0, 70], [0, 104], [109, 121], [130, 255]]
NAMES = {1: "pore", 2: "dark phase", 3: "grey phase", 4: "bright phase"}
log = {}


def step(m):
    print(f"· {m}", flush=True)


# 1 ── data health -----------------------------------------------------------------
step("data health")
raw = auto_crop(Volume.from_zarr("projects/fib_sem_335/raw.zarr"))
qc = inspect(raw)
for k, df in qc.items():
    df.to_csv(T / f"beyond_qc_{k}.csv", index=False)
s, fl, summ = qc["slices"], qc["flags"], qc["summary"].iloc[0]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    for a, (col, lab) in zip(ax, [("mean", "brightness"), ("snr", "SNR"),
                                  ("sharpness", "sharpness"), ("curtaining", "curtaining index")]):
        a.plot(s.z, s[col], lw=0.8, color=PALETTE[1])
        f = fl[fl.metric == col]
        a.scatter(f.z, s[col].iloc[f.z], color=PALETTE[5], s=22, zorder=3,
                  label="flagged" if len(f) else None)
        a.set_ylabel(lab)
        if len(f):
            a.legend(frameon=False, loc="upper right")
    ax[-1].set_xlabel("slice")
    fig.suptitle(f"Data health — {int(summ.flagged_slices)} of {int(summ.slices)} slices flagged; "
                 f"median SNR {summ.snr_median:.1f}; brightness drift {summ.brightness_drift_pct:+.1f} %; "
                 f"correlation length Z/XY = {summ.z_over_xy:.2f}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "16_data_health.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
log["qc"] = summ.to_dict()

# 2 ── Noise2Void vs NLM ------------------------------------------------------------
step("Noise2Void")
corrected = Volume.from_zarr(V / "corrected.zarr")
binned = bin_volume(corrected, (1, 2, 2))
t = time.time()
n2v = denoise_n2v(binned, steps=2000, model_path=str(PROJ / "n2v_335.pt")).to_zarr(
    V / "denoised_n2v.zarr")
log["n2v_seconds"] = round(time.time() - t, 1)
nlm = Volume.from_zarr(V / "denoised.zarr")
z = 400
crop = (slice(140, 300), slice(260, 470))
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, (name, im) in zip(ax, [("binned input", binned.get_slice(0, z)),
                                  ("non-local means", nlm.get_slice(0, z)),
                                  ("Noise2Void (self-supervised)", n2v.get_slice(0, z))]):
        a.imshow(im[crop], cmap="gray", vmin=70, vmax=170, interpolation="nearest")
        a.set_title(name, loc="left"), a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Denoising, slice {z} (1.6 × 2.1 µm). Noise2Void trained on this stack in "
                 f"{log['n2v_seconds']:.0f} s on the GPU, no clean reference needed",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "17_denoise_n2v.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 3 ── segmentations on N2V input: random forest and U-Net ------------------------------
step("random forest on N2V")
ann = auto_annotations(n2v, bands=BANDS, n_slices=10)
rf = PixelClassifier(n_estimators=60, class_names=NAMES).fit(n2v, ann).predict(n2v)
rf = smooth_labels(rf).to_zarr(V / "phases_n2v_rf.zarr")
step("U-Net on N2V")
t = time.time()
un = unet_segment(n2v, annotations=ann, classes=4, class_names=NAMES, steps=2500,
                  model_path=str(PROJ / "unet_335.pt"))["volume"]
un = smooth_labels(un).to_zarr(V / "phases_n2v_unet.zarr")
log["unet_seconds"] = round(time.time() - t, 1)
base = Volume.from_zarr(V / "phases.zarr").numpy()
rows = []
for name, vol in [("NLM + RF (recipe)", base), ("N2V + RF", rf.numpy()),
                  ("N2V + U-Net", un.numpy())]:
    r = {"pipeline": name, "agreement_with_recipe": float((vol == base).mean())}
    for c, n in NAMES.items():
        r[n] = float((vol == c).mean())
    rows.append(r)
seg = pd.DataFrame(rows)
seg.to_csv(T / "beyond_segmentation_compare.csv", index=False)
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    ax[0].imshow(n2v.get_slice(0, z)[crop], cmap="gray", vmin=70, vmax=170)
    ax[0].set_title("Noise2Void input", loc="left")
    for a, (name, vol) in zip(ax[1:], [("NLM + random forest", base), ("N2V + random forest",
                                                                     rf.numpy()),
                                       ("N2V + U-Net (2.5D)", un.numpy())]):
        a.imshow(n2v.get_slice(0, z)[crop], cmap="gray", vmin=70, vmax=170)
        a.imshow(vol[z][crop], cmap=label_cmap(4, 0.55), vmin=0, vmax=4, interpolation="nearest")
        a.set_title(name, loc="left")
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "18_segmentation_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 4 ── confidence: RVE + uncertainty ------------------------------------------------
step("RVE and uncertainty")
phases = Volume.from_zarr(V / "phases.zarr")
rv = rve(phases, phases=[2, 3, 4])
rv["table"].to_csv(T / "beyond_rve.csv", index=False)
rv["curves"].to_csv(T / "beyond_rve_curves.csv", index=False)
un_t = uncertainty(phases, phases=[2, 3, 4], axis=0)
un_t.to_csv(T / "beyond_uncertainty.csv", index=False)
tab = rv["table"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    vf = un_t[un_t.metric == "volume_fraction"].reset_index(drop=True)
    x = np.arange(len(vf))
    cols = [PALETTE[(c - 1) % 10] for c in vf.phase]
    ax[0].bar(x, 100 * vf.value, color=cols, alpha=0.85)
    ax[0].errorbar(x - 0.12, 100 * vf.value, yerr=[100 * (vf.value - vf.sampling_ci95_low),
                                                   100 * (vf.sampling_ci95_high - vf.value)],
                   fmt="none", ecolor="white", capsize=6, lw=2, label="sampling (95 %)")
    ax[0].errorbar(x + 0.12, 100 * vf.value, yerr=[100 * (vf.value - vf.boundary_low),
                                                   100 * (vf.boundary_high - vf.value)],
                   fmt="none", ecolor=PALETTE[0], capsize=6, lw=2, label="boundary ±1 voxel")
    ax[0].set_xticks(x, vf.name), ax[0].set_ylabel("volume %")
    ax[0].set_title("Volume fractions with uncertainty", loc="left")
    ax[0].legend(frameon=False)
    for c, g in rv["curves"].groupby("phase"):
        p = tab.set_index("phase").loc[c, "fraction"]
        ax[1].loglog(g.volume_um3, np.sqrt(g["var"]) / p, "o-", ms=3,
                     color=PALETTE[(c - 1) % 10], label=g.name.iloc[0])
    ax[1].axhline(0.025, color="#8a94a6", ls="--", lw=1)
    ax[1].text(ax[1].get_xlim()[0] * 1.2, 0.027, "±5 % (95 %)", color="#8a94a6", fontsize=9)
    ax[1].set_xlabel("sub-volume (µm³)"), ax[1].set_ylabel("relative std of fraction")
    ax[1].set_title("Fraction scatter vs sampled volume", loc="left")
    ax[1].legend(frameon=False)
    col = [c for c in tab.columns if c.startswith("rve_edge")][0]
    ax[2].barh(tab.name, tab[col], color=[PALETTE[(c - 1) % 10] for c in tab.phase])
    edge = tab.volume_um3.iloc[0] ** (1 / 3)
    ax[2].axvline(edge, color="white", ls="--", lw=1.2)
    ax[2].text(edge * 1.03, -0.4, f"this volume\n≈ {edge:.1f} µm cube", color="white",
               fontsize=9, va="bottom")
    ax[2].set_xlabel("cube edge for ±5 % (µm)")
    ax[2].set_title("Representative volume needed", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "19_confidence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

(T / "beyond_log.json").write_text(json.dumps(log, indent=2, default=str))
print(seg.round(4).to_string())
print(tab.round(3).to_string())
print(un_t.round(3).to_string())
print("done", log)
