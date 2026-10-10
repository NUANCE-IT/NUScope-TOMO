"""Acquisition advice for sample 335 from its first 100 slices and from all 812.

    python examples/advisor_demo.py [project_dir]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nuscope_tomo import Volume  # noqa: E402
from nuscope_tomo.advisor import advise  # noqa: E402
from nuscope_tomo.viz.figures import STYLE  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
ph = Volume.from_zarr(PROJ / "volumes" / "phases.zarr")
res = {n: advise(ph.roi(slice(0, n)), phases=[2, 3, 4]) for n in (100, 812)}
rows = []
for n, r in res.items():
    rec = r["recommendation"].iloc[0].to_dict()
    rec["slices_seen"] = n
    rows.append(rec)
    r["coarsening"].assign(slices_seen=n).to_csv(PROJ / "tables" / f"advisor_coarsening_{n}.csv",
                                                 index=False)
    r["volume"].assign(slices_seen=n).to_csv(PROJ / "tables" / f"advisor_volume_{n}.csv", index=False)
pd.DataFrame(rows).to_csv(PROJ / "tables" / "advisor.csv", index=False)
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, (n, r) in zip(ax, res.items()):
        c = r["coarsening"]
        px, sl = sorted(c.pixel_nm.unique()), sorted(c.slice_nm.unique())
        M = c.pivot(index="pixel_nm", columns="slice_nm", values="max_rel_change").loc[px, sl]
        im = a.imshow(100 * M.to_numpy(), cmap="magma_r", vmin=0, vmax=30, origin="lower")
        for i in range(len(px)):
            for j in range(len(sl)):
                v = 100 * M.iloc[i, j]
                a.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                       color="black" if v < 15 else "white")
        rec = r["recommendation"].iloc[0]
        j, i = sl.index(rec.recommended_slice_nm), px.index(rec.recommended_pixel_nm)
        a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#3ddc97", lw=3))
        a.set_xticks(range(len(sl)), [f"{v:.0f}" for v in sl])
        a.set_yticks(range(len(px)), [f"{v:.0f}" for v in px])
        a.set_xlabel("slice thickness (nm)"), a.set_ylabel("pixel size (nm)")
        a.set_title(f"after {n} slices: largest change in any result", loc="left")
        plt.colorbar(im, ax=a, fraction=0.04, label="%")
    fig.suptitle("Acquisition advisor: how much would coarser sampling have changed the answer? "
                 "(green = recommended, ≤ 5 %)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PROJ / "gallery" / "23_advisor.png", dpi=130, bbox_inches="tight")
for r in rows:
    print(r["slices_seen"], r["advice"])
