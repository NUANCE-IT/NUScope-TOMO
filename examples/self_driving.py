"""Self-driving slice-and-view on a digital twin of sample 335.

    python examples/self_driving.py agent|conventional [project_dir]

The twin is built from the measured stack (Noise2Void-cleaned) and its
segmentation. The agent runs through ShadowAutoScript, so every Thermo Fisher
AutoScript call it would send to a real DualBeam is recorded (dry run).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from nuscope_tomo import Volume
from nuscope_tomo.autonomy import (DigitalTwin, Policy, ShadowAutoScript, SliceAndViewAgent,
                                   conventional_policy)

mode = sys.argv[1]
PROJ = Path(sys.argv[2] if len(sys.argv) > 2 else "projects/fib_sem_335/recipe_run")
V, OUT = PROJ / "volumes", PROJ / "autonomy"
OUT.mkdir(exist_ok=True)
clean, ph = Volume.from_zarr(V / "denoised_n2v.zarr"), Volume.from_zarr(V / "phases.zarr")
clf = pickle.loads(Path("projects/fib_sem_335/models/rf_335.pkl").read_bytes())
COMP = {1: {"C": .7, "O": .2, "Ni": .05, "Mn": .05}, 2: {"Ni": .15, "Mn": .55, "Co": .05, "O": .25},
        3: {"Ni": .45, "Mn": .15, "Co": .15, "O": .25}, 4: {"Ni": .60, "Mn": .05, "Co": .10, "O": .25}}
twin = DigitalTwin(clean, ph, COMP, noise_at_1us=10, seed=1)
common = dict(max_depth_nm=16000, eds_every=25, target_rel_error=0.20,
              precision_phases=["dark phase", "grey phase"], stats_every=10)
if mode == "agent":
    inst = ShadowAutoScript(twin)
    pol = Policy(max_slice_nm=60, **common)
elif mode == "conventional_stop":  # fixed recipe + the agent's stop-at-precision rule only
    inst = twin
    pol = conventional_policy(dwell_us=3.0, **common)
    pol.stop_on_precision = True
else:
    inst = twin
    pol = conventional_policy(dwell_us=3.0, **common)
t = time.time()
res = SliceAndViewAgent(inst, clf, pol, log=print).run()
res["trace"].to_csv(OUT / f"{mode}_trace.csv", index=False)
res["decisions"].to_csv(OUT / f"{mode}_decisions.csv", index=False)
if mode == "agent":
    pd.DataFrame(inst.rec.calls.entries).to_csv(OUT / "autoscript_calls.csv", index=False)
summary = {k: res[k] for k in ("reached", "time_h", "depth_um", "slices", "ledger_h",
                                "eds_pixels", "eds_pixels_full")}
summary["final"] = {k: {"fraction": v[0], "rel_error": v[1]} for k, v in res["final"].items()}
summary["wall_s"] = time.time() - t
(OUT / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
