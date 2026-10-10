"""Figure for the self-driving ablation (run examples/self_driving.py first for
conventional, conventional_stop and agent)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nuscope_tomo.viz import PALETTE  # noqa: E402
from nuscope_tomo.viz.figures import STYLE  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
A = PROJ / "autonomy"
names = {"conventional": "fixed recipe (full planned depth)",
         "conventional_stop": "fixed recipe + stop at precision",
         "agent": "self-driving agent"}
S = {m: json.loads((A / f"{m}_summary.json").read_text()) for m in names}
Tr = {m: pd.read_csv(A / f"{m}_trace.csv") for m in names}
dec = pd.read_csv(A / "agent_decisions.csv")
rows = []
for m, s in S.items():
    rows.append({"policy": names[m], "time_h": s["time_h"], "depth_um": s["depth_um"],
                 "slices": s["slices"], "eds_pixels": s["eds_pixels"],
                 **{f"err_{k}": v["rel_error"] for k, v in s["final"].items()},
                 **{f"t_{k}": v for k, v in s["ledger_h"].items()}})
tab = pd.DataFrame(rows)
tab.to_csv(PROJ / "tables" / "self_driving_ablation.csv", index=False)
print(tab.round(3).to_string())
cols = {"mill": PALETTE[4], "overhead": PALETTE[6], "image": PALETTE[1], "eds": PALETTE[0],
        "autofocus": PALETTE[2]}
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.2), gridspec_kw={"width_ratios": [1.1, 1.2, 1.3]})
    left = 0
    for i, m in enumerate(names):
        acc = 0
        for k, c in cols.items():
            v = S[m]["ledger_h"].get(k, 0)
            ax[0].barh(i, v, left=acc, color=c, label=k if i == 0 else None)
            acc += v
        ax[0].text(acc + 0.1, i, f"{acc:.1f} h · {S[m]['depth_um']:.1f} µm · {S[m]['slices']} slices",
                   va="center", fontsize=9)
    ax[0].set_yticks(range(3), [names[m] for m in names], fontsize=9)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("instrument time (h)")
    ax[0].set_xlim(0, max(s["time_h"] for s in S.values()) * 1.6)
    ax[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax[0].set_title("Where the time goes", loc="left")
    for i, m in enumerate(names):
        e = Tr[m].dropna(subset=["err_dark phase"])
        ax[1].plot(e.t_h, 100 * e["err_dark phase"], color=["#8a94a6", PALETTE[1], PALETTE[0]][i], lw=2.2,
                   label=names[m])
    ax[1].axhline(20, color="#8a94a6", ls="--", lw=1)
    ax[1].text(ax[1].get_xlim()[1] * 0.02 if False else 0.1, 21, "target ±20 %", color="#8a94a6",
               fontsize=9)
    ax[1].set_xlabel("instrument time (h)"), ax[1].set_ylabel("dark-phase fraction error, 95 % (%)")
    ax[1].set_title("Precision as the run proceeds", loc="left")
    ax[1].legend(frameon=False, fontsize=9)
    tr = Tr["agent"]
    ax[2].step(tr.depth_um, tr.thickness_nm, where="post", color=PALETTE[4], label="slice thickness (nm)")
    ax2 = ax[2].twinx()
    ax2.plot(tr.depth_um, tr.dwell_us, color=PALETTE[2], ls="--", label="dwell (µs)")
    ax2.set_ylabel("dwell (µs)", color=PALETTE[2])
    ax[2].set_ylabel("slice thickness (nm)", color=PALETTE[4])
    marks = {"autofocus": ("v", PALETTE[2]), "beam_shift": ("|", "#cccccc"), "eds": ("D", PALETTE[0]),
             "stop": ("X", PALETTE[5])}
    ymax = tr.thickness_nm.max() * 1.25
    for k, (mk, c) in marks.items():
        d = dec[dec.action == k]
        ax[2].scatter(d.depth_um, [ymax] * len(d), marker=mk, color=c, s=40, label=k, zorder=3)
    ax[2].set_ylim(0, ymax * 1.12)
    ax[2].set_xlabel("milled depth (µm)")
    ax[2].legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    ax[2].set_title("The agent's decisions (AutoScript calls recorded in shadow mode)", loc="left",
                    fontsize=11)
    fig.suptitle("Self-driving slice-and-view on a digital twin of sample 335: same ±20 % precision on the "
                 "dark and grey phase fractions", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PROJ / "gallery" / "25_self_driving.png", dpi=125, bbox_inches="tight")
