"""Context-driven sparse EDS on sample 335 (simulated EDS on the real segmentation).

    python examples/sparse_eds.py [project_dir]

Deep kernel learning learns SE context -> composition from a few measured EDS
pixels and chooses the next ones where it is least certain (Kalinin/Ziatdinov-style
active learning). Compared with full maps, random sampling and interpolation.
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nuscope_tomo import Volume  # noqa: E402
from nuscope_tomo.chem import simulate_eds  # noqa: E402
from nuscope_tomo.sparse import active_sparse, reconstruct_interpolate  # noqa: E402
from nuscope_tomo.viz import PALETTE  # noqa: E402
from nuscope_tomo.viz.figures import STYLE  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
V, T, OUT = PROJ / "volumes", PROJ / "tables", PROJ / "gallery"
se, ph = Volume.from_zarr(V / "denoised_n2v.zarr"), Volume.from_zarr(V / "phases.zarr")
COMP = {1: {"C": .7, "O": .2, "Ni": .05, "Mn": .05}, 2: {"Ni": .15, "Mn": .55, "Co": .05, "O": .25},
        3: {"Ni": .45, "Mn": .15, "Co": .15, "O": .25}, 4: {"Ni": .60, "Mn": .05, "Co": .10, "O": .25}}
B = 6
rows, keep = [], {}
for z in (200, 400, 600):
    lab = ph.numpy()[z:z + 1]
    img = se.get_slice(0, z).astype(np.float32)
    eds, tr = simulate_eds(lab, COMP, ph.voxel_size, bin_xy=B, every=1, counts_per_pixel=40,
                           shift_px=(0, 0), psf_nm=60, seed=z)
    els = tr["elements"]
    truth = np.stack([tr["noise_free"][e][0] for e in els], -1)
    truth /= truth.sum(-1, keepdims=True)
    cnt = np.stack([eds.maps[e][0] for e in els], -1)
    meas = cnt / np.maximum(cnt.sum(-1, keepdims=True), 1)
    ny, nx = truth.shape[:2]
    flat = meas.reshape(-1, len(els))
    cent = np.array([[COMP[c].get(e, 0) for e in els] for c in (1, 2, 3, 4)])
    cent /= cent.sum(1, keepdims=True)
    blk = lab[0][:ny * B, :nx * B].reshape(ny, B, nx, B).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    maj = np.apply_along_axis(lambda v: np.bincount(v, minlength=5).argmax(), 2, blk)

    def score(rec):
        cls = np.argmin(((rec[..., None, :] - cent) ** 2).sum(-1), -1) + 1
        return float(np.sqrt(np.mean((rec - truth) ** 2))), float((cls == maj).mean())

    rows.append({"slice": z, "method": "full map", "budget": 1.0, **dict(zip(("rmse", "acc"), score(meas)))})
    for b in (0.02, 0.05, 0.10):
        for strat in ("variance", "random"):
            t = time.time()
            r = active_sparse(img, (ny, nx), B, lambda i: flat[i], budget=b, init=b / 4, rounds=3,
                              strategy=strat, seed=z)
            name = "DKL active" if strat == "variance" else "DKL random"
            rows.append({"slice": z, "method": name, "budget": b, "seconds": time.time() - t,
                         **dict(zip(("rmse", "acc"), score(r["mean"])))})
            if strat == "random":
                itp = reconstruct_interpolate((ny, nx), r["measured_idx"], r["measured_values"])
                rows.append({"slice": z, "method": "interpolation", "budget": b,
                             **dict(zip(("rmse", "acc"), score(itp)))})
            if z == 400 and b == 0.05:
                keep[name] = r
                if strat == "random":
                    keep["interp"] = itp
    if z == 400:
        keep.update(truth=truth, meas=meas, els=els, img=img)
    print("slice", z, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(T / "sparse_eds.csv", index=False)
agg = df.groupby(["method", "budget"])[["rmse", "acc"]].mean().reset_index()
print(agg.round(4).to_string())
mn = keep["els"].index("Mn")
with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(21, 8.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.28, wspace=0.08)
    panels = [("SE context (Noise2Void)", keep["img"], "gray", None),
              ("true Mn fraction", keep["truth"][..., mn], "magma", (0, .6)),
              ("full EDS map (100 % pixels)", keep["meas"][..., mn], "magma", (0, .6)),
              ("interpolation, 5 % random", keep["interp"][..., mn], "magma", (0, .6)),
              ("DKL, 5 % actively chosen", keep["DKL active"]["mean"][..., mn], "magma", (0, .6)),
              ("DKL uncertainty + measured pixels", keep["DKL active"]["variance"], "viridis", None)]
    for i, (t, im, cm, cl) in enumerate(panels):
        a = fig.add_subplot(gs[0, i])
        a.imshow(im, cmap=cm, vmin=None if cl is None else cl[0], vmax=None if cl is None else cl[1],
                 interpolation="nearest")
        if i == 5:
            yy, xx = np.nonzero(keep["DKL active"]["measured_mask"])
            a.scatter(xx, yy, s=3, c="white", linewidths=0)
        a.set_title(t, loc="left", fontsize=10)
        a.set_xticks([]), a.set_yticks([])
    styles = {"full map": ("white", "--"), "interpolation": (PALETTE[5], "-"),
              "DKL random": (PALETTE[1], "-"), "DKL active": (PALETTE[0], "-")}
    for j, (col, lab_) in enumerate((("rmse", "composition RMSE"), ("acc", "phase accuracy"))):
        a = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        for m, (c, ls) in styles.items():
            g = agg[agg.method == m]
            if m == "full map":
                a.axhline(g[col].iloc[0], color=c, ls=ls, lw=1.2, label="full map (100 % pixels)")
            else:
                a.plot(100 * g.budget, g[col], "o-", color=c, ls=ls, label=m)
        a.set_xlabel("EDS pixels measured (%)"), a.set_ylabel(lab_)
        a.legend(frameon=False, fontsize=9)
    fig.suptitle("Context-driven sparse EDS (deep kernel learning, after Kalinin & Ziatdinov) — "
                 "SIMULATED EDS on the real sample 335 segmentation, mean of 3 slices",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "22_sparse_eds.png", dpi=120, bbox_inches="tight")
