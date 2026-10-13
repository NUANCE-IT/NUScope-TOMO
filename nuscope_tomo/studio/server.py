"""NUScope-TOMO Studio backend: serves a project's volumes, tables and renders to
the browser UI, and runs registered operations as background jobs.

Data never leaves the machine running the server; the browser only receives
2D slices (PNG) and a downsampled 3D preview (raw uint8) on demand, so a
workstation or HPC node can host the data while any laptop drives the UI.
"""
from __future__ import annotations

import io
import json
import pickle
import shutil
import threading
import time
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.registry import get_op, list_ops
from ..core.volume import META_FILE, Volume

STATIC = Path(__file__).parent / "static"

# Simulated chemistry for the twin's EDS (NMC-like phases of sample 335); replace with real
# compositions when EDS reference spectra are available.
DEFAULT_COMP = {1: {"C": .7, "O": .2, "Ni": .05, "Mn": .05},
                2: {"Ni": .15, "Mn": .55, "Co": .05, "O": .25},
                3: {"Ni": .45, "Mn": .15, "Co": .15, "O": .25},
                4: {"Ni": .60, "Mn": .05, "Co": .10, "O": .25}}

# Guided pre-processing for slice-and-view stacks (cf. Avizo "DualBeam 3D Wizard").
WIZARD_STEPS = [
    {"op": "preprocess.auto_crop", "title": "Find block face",
     "help": "Remove Pt cap, trench walls and padding", "params": {"margin": 20}},
    {"op": "preprocess.align", "title": "Align slices",
     "help": "Sub-pixel drift correction; skipped if already registered", "params": {}},
    {"op": "preprocess.destripe", "title": "Remove curtaining",
     "help": "Wavelet–FFT stripe filter", "params": {"level": 5, "sigma": 3.0}},
    {"op": "preprocess.flatten_shading", "title": "Flatten shading",
     "help": "Divide out the slow illumination gradient", "params": {}},
    {"op": "preprocess.normalize_slices", "title": "Even out slice brightness",
     "help": "Match each slice to a running median", "params": {"window": 15}},
    {"op": "preprocess.bin", "title": "Bin XY", "help": "Average 2×2 pixels",
     "params": {"factors": [1, 2, 2]}},
    {"op": "preprocess.denoise_nlm", "title": "Denoise", "help": "Non-local means per slice",
     "params": {"h": 0.8}},
]


class Project:
    """Index of the volumes / tables / media found under a project folder."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.memory: dict[str, Volume] = {}  # results created in this session
        self.refresh()

    def refresh(self):
        vols: dict[str, Path] = {}
        for p in sorted(self.root.rglob("*.zarr")):
            if (p / "zarr.json").exists() and not p.name.startswith("_") \
                    and not any(q.suffix == ".zarr" for q in p.parents):
                vols[str(p.relative_to(self.root).with_suffix(""))] = p
        self.volumes = vols  # swap atomically: concurrent requests never see a partial index
        self.tables = {str(p.relative_to(self.root)): p for p in sorted(self.root.rglob("*.csv"))}
        self.media = {str(p.relative_to(self.root)): p for ext in ("png", "mp4", "jpg")
                      for p in sorted(self.root.rglob(f"*.{ext}"))}

    def volume(self, vid: str) -> Volume:
        if vid in self.memory:
            return self.memory[vid]
        if vid not in self.volumes:
            raise HTTPException(404, f"no volume {vid}")
        if vid.startswith("live/"):  # growing during acquisition: reopen every time
            return Volume.from_zarr(self.volumes[vid])
        return _open(str(self.volumes[vid]))


@lru_cache(maxsize=32)
def _open(path: str) -> Volume:
    return Volume.from_zarr(path)


def _png(img: np.ndarray) -> bytes:
    import imageio.v3 as iio
    return iio.imwrite("<bytes>", img, extension=".png")


def _window(vol: Volume) -> tuple[float, float]:
    w = vol.meta.get("_studio_window")
    if w:
        return w
    sub = np.asarray(vol.data[:: max(1, vol.shape[0] // 16), ::4, ::4])
    nz = sub[sub > 0] if (sub > 0).mean() > 0.5 else sub.ravel()
    lo, hi = np.percentile(nz, (0.5, 99.5)) if nz.size else (0, 1)
    vol.meta["_studio_window"] = (float(lo), float(max(hi, lo + 1e-6)))
    return vol.meta["_studio_window"]


def _to_u8(a: np.ndarray, vol: Volume) -> np.ndarray:
    if vol.kind == "labels":
        if a.max() > 255:  # object ids: fold into 1..255 for display colouring
            a = np.where(a > 0, (a - 1) % 255 + 1, 0)
        return a.astype(np.uint8)
    if a.dtype == np.uint8 and vol.meta.get("_studio_raw8", True):
        return a
    lo, hi = _window(vol)
    return np.clip((a.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def create_app(project_dir: str | Path) -> FastAPI:
    app = FastAPI(title="NUScope-TOMO Studio", version=__version__)
    proj = Project(project_dir)
    jobs: dict[str, dict] = {}

    @app.get("/api/project")
    def project():
        proj.refresh()
        vols = []
        for vid in list(proj.volumes) + list(proj.memory):
            try:
                v = proj.volume(vid)
            except Exception:
                continue
            vols.append({"id": vid, "name": v.name, "kind": v.kind, "shape": v.shape,
                         "dtype": str(v.dtype), "voxel_nm": v.voxel_size.as_tuple(),
                         "classes": v.meta.get("classes", {}),
                         "in_memory": vid in proj.memory})
        return {"root": str(proj.root), "name": proj.root.name, "version": __version__,
                "volumes": vols, "tables": list(proj.tables), "media": list(proj.media),
                "mtime": {t: proj.tables[t].stat().st_mtime for t in proj.tables},
                "models": sorted(str(p.relative_to(proj.root)) for p in proj.root.rglob("*.pkl"))}

    @app.get("/api/volume/{vid:path}/info")
    def info(vid: str):
        v = proj.volume(vid)
        h, e = v.histogram(bins=128, step=max(1, int(round((v.data.size / 4e6) ** (1 / 3)))))
        meta = {k: val for k, val in v.meta.items()
                if not k.startswith("_") and (not isinstance(val, list) or len(val) < 50)}
        return {"id": vid, "name": v.name, "kind": v.kind, "shape": v.shape,
                "dtype": str(v.dtype), "voxel_nm": v.voxel_size.as_tuple(),
                "extent_um": [x / 1e3 for x in v.extent_nm], "meta": json.loads(
                    json.dumps(meta, default=str)),
                "histogram": {"counts": h.tolist(), "edges": e.tolist()}}

    @app.get("/api/volume/{vid:path}/slice")
    def slice_(vid: str, axis: int = 0, index: int = 0, lo: int = 0, hi: int = 255):
        v = proj.volume(vid)
        img = _to_u8(v.get_slice(axis, index), v)
        if v.kind != "labels" and (lo, hi) != (0, 255):
            img = np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255
                          ).astype(np.uint8)
        return Response(_png(img), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/volume/{vid:path}/value")
    def value(vid: str, z: int, y: int, x: int):
        v = proj.volume(vid)
        z, y, x = (int(np.clip(c, 0, n - 1)) for c, n in zip((z, y, x), v.shape))
        val = np.asarray(v.data[z, y, x]).item()
        name = v.meta.get("classes", {}).get(str(int(val))) if v.kind == "labels" else None
        return {"value": val, "name": name, "z": z, "y": y, "x": x}

    @app.get("/api/volume/{vid:path}/preview3d")
    def preview3d(vid: str, max_voxels: int = 160**3):
        v = proj.volume(vid)
        pv = v.preview(max_voxels=max_voxels, isotropic=True)
        arr = _to_u8(pv.numpy(), v)
        hdr = json.dumps({"shape": arr.shape, "voxel_nm": pv.voxel_size.as_tuple(),
                          "kind": v.kind}).encode()
        return Response(len(hdr).to_bytes(4, "little") + hdr + arr.tobytes(),
                        media_type="application/octet-stream")

    @app.get("/api/table/{tid:path}")
    def table(tid: str, limit: int = 5000):
        if tid not in proj.tables:
            raise HTTPException(404)
        df = pd.read_csv(proj.tables[tid]).head(limit)
        return JSONResponse(json.loads(df.to_json(orient="split", index=False)))

    @app.get("/api/media/{mid:path}")
    def media(mid: str):
        if mid not in proj.media:
            raise HTTPException(404)
        return FileResponse(proj.media[mid])

    @app.get("/api/ops")
    def ops():
        return [{"name": n, "summary": s.summary, "category": s.category,
                 "inputs": list(s.inputs), "params": json.loads(json.dumps(s.params(), default=str))}
                for n, s in list_ops().items()]

    @app.post("/api/annotations/{vid:path}")
    def save_annotations(vid: str, strokes: list = Body(...)):
        out = proj.root / "annotations" / f"{vid.replace('/', '__')}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(strokes))
        return {"saved": str(out.relative_to(proj.root)), "n": len(strokes)}

    @app.get("/api/annotations/{vid:path}")
    def load_annotations(vid: str):
        p = proj.root / "annotations" / f"{vid.replace('/', '__')}.json"
        return json.loads(p.read_text()) if p.exists() else []

    @app.post("/api/run")
    def run(req: dict = Body(...)):
        """{"op": name, "inputs": {param: volume_id}, "params": {...}, "save": bool}"""
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {"id": jid, "op": req["op"], "status": "running", "t0": time.time(),
                     "log": []}

        def work():
            try:
                spec = get_op(req["op"])
                kw = dict(req.get("params", {}))
                for name, vid in req.get("inputs", {}).items():
                    kw[name] = proj.volume(vid)
                if kw.get("scribbles") == "@annotations":
                    first = next(iter(req.get("inputs", {}).values()))
                    kw["scribbles"] = str(proj.root / "annotations" /
                                          f"{first.replace('/', '__')}.json")
                out = spec.func(**kw)
                results = out if isinstance(out, dict) else {"result": out}
                made = []
                for k, val in results.items():
                    rid = f"studio/{req['op'].split('.')[-1]}_{jid}" + ("" if k in ("volume", "result") else f"_{k}")
                    if isinstance(val, Volume):
                        if req.get("save", True):
                            val = val.to_zarr(proj.root / f"{rid}.zarr")
                            _open.cache_clear()
                        else:
                            proj.memory[rid] = val
                        made.append(rid)
                    elif isinstance(val, pd.DataFrame):
                        p = proj.root / f"{rid}.csv"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        val.to_csv(p, index=False)
                        made.append(str(p.relative_to(proj.root)))
                _log_recipe(proj.root, req)
                jobs[jid].update(status="done", outputs=made)
            except Exception as e:  # surfaced to the UI
                jobs[jid].update(status="error", error=f"{e}", trace=traceback.format_exc())
            jobs[jid]["seconds"] = round(time.time() - jobs[jid]["t0"], 2)

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    @app.get("/api/wizard/steps")
    def wizard_steps():
        return WIZARD_STEPS

    @app.post("/api/wizard")
    def wizard(req: dict = Body(...)):
        """Guided FIB-SEM pre-processing: {"volume": id, "steps": [{op, params, enabled}]}."""
        jid = uuid.uuid4().hex[:8]
        steps = [s for s in req.get("steps", WIZARD_STEPS) if s.get("enabled", True)]
        jobs[jid] = {"id": jid, "op": "FIB-SEM wizard", "status": "running", "t0": time.time(),
                     "log": [{"op": s["op"], "title": s.get("title", s["op"]), "status": "pending"}
                             for s in steps]}

        def work():
            try:
                vol = proj.volume(req["volume"])
                for i, st in enumerate(steps):
                    entry = jobs[jid]["log"][i]
                    entry["status"] = "running"
                    t = time.time()
                    out = get_op(st["op"]).func(vol, **st.get("params", {}))
                    vol = out["volume"] if isinstance(out, dict) else out
                    if st["op"] in ("preprocess.destripe", "preprocess.normalize_slices"):
                        # persist after expensive lazy stages so later steps don't recompute
                        vol = vol.to_zarr(proj.root / "studio" / f"_wizard_{jid}_{i}.zarr")
                        _open.cache_clear()
                    entry.update(status="done", seconds=round(time.time() - t, 1),
                                 shape=list(vol.shape),
                                 note=str(vol.meta.get("aligned", "")) if st["op"] ==
                                 "preprocess.align" else "")
                    _log_recipe(proj.root, {"op": st["op"], "inputs": {"volume": req["volume"]},
                                            "params": st.get("params", {})})
                rid = f"studio/wizard_{jid}"
                vol.to_zarr(proj.root / f"{rid}.zarr")
                _open.cache_clear()
                for tmp in (proj.root / "studio").glob(f"_wizard_{jid}_*.zarr"):
                    shutil.rmtree(tmp, ignore_errors=True)
                jobs[jid].update(status="done", outputs=[rid])
            except Exception as e:
                jobs[jid].update(status="error", error=str(e), trace=traceback.format_exc())
            jobs[jid]["seconds"] = round(time.time() - jobs[jid]["t0"], 2)

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    @app.get("/api/live")
    def live():
        p = proj.root / "live" / "live.json"
        if not p.exists():
            return {"status": "none"}
        doc = json.loads(p.read_text())
        doc["age_s"] = round(time.time() - p.stat().st_mtime, 1)
        return doc

    @app.get("/api/jobs")
    def list_jobs():
        return sorted(({k: v for k, v in j.items() if not k.startswith("_")} for j in jobs.values()),
                      key=lambda j: -j["t0"])

    # ------------------------------------------------------------ self-driving (twin)
    agents: dict[str, dict] = {}
    stop_file = proj.root / "autonomy" / "STOP"

    @app.post("/api/agent/start")
    def agent_start(req: dict = Body(...)):
        """Run the slice-and-view agent on a digital twin of the project's data.

        {"policy": "agent"|"fixed"|"fixed_stop", "clean": vid, "labels": vid, "model": path,
         "target": 0.2, "max_depth_um": 16, "phases": [names], "eds_every": 25}
        """
        from ..autonomy import (DigitalTwin, Policy, SafetyLimits, ShadowAutoScript,
                                SliceAndViewAgent, conventional_policy)
        mode = req.get("policy", "agent")
        jid = uuid.uuid4().hex[:8]
        stop_file.parent.mkdir(exist_ok=True)
        stop_file.unlink(missing_ok=True)
        names = {"agent": "self-driving agent", "fixed": "fixed recipe",
                 "fixed_stop": "fixed recipe + stop"}
        jobs[jid] = {"id": jid, "op": f"twin run: {names.get(mode, mode)}", "status": "running",
                     "t0": time.time(), "kind": "agent", "policy": mode}

        def work():
            try:
                clean, labels = proj.volume(req["clean"]), proj.volume(req["labels"])
                clf = pickle.loads((proj.root / req["model"]).read_bytes())
                comp = {int(k): v for k, v in (req.get("compositions") or DEFAULT_COMP).items()}
                twin = DigitalTwin(clean, labels, comp, limits=SafetyLimits(stop_file=stop_file),
                                   noise_at_1us=10, seed=1)
                common = dict(max_depth_nm=float(req.get("max_depth_um", 16)) * 1e3,
                              eds_every=int(req.get("eds_every", 25)),
                              target_rel_error=float(req.get("target", 0.2)),
                              precision_phases=req.get("phases") or None, stats_every=10)
                if mode == "agent":
                    inst, pol = ShadowAutoScript(twin), Policy(max_slice_nm=60, **common)
                else:
                    inst, pol = twin, conventional_policy(dwell_us=3.0, **common)
                    pol.stop_on_precision = mode == "fixed_stop"
                ag = SliceAndViewAgent(inst, clf, pol)
                agents[jid] = {"agent": ag, "inst": inst}
                res = ag.run()
                out = proj.root / "autonomy"
                res["trace"].to_csv(out / f"{mode}_{jid}_trace.csv", index=False)
                res["decisions"].to_csv(out / f"{mode}_{jid}_decisions.csv", index=False)
                made = [f"autonomy/{mode}_{jid}_trace.csv", f"autonomy/{mode}_{jid}_decisions.csv"]
                if hasattr(inst, "rec"):
                    pd.DataFrame(inst.rec.calls.entries).to_csv(out / f"autoscript_{jid}.csv",
                                                                index=False)
                    made.append(f"autonomy/autoscript_{jid}.csv")
                jobs[jid].update(status="done", outputs=made, summary={
                    "reached": res["reached"], "time_h": res["time_h"], "depth_um": res["depth_um"],
                    "slices": res["slices"], "eds_pixels": res["eds_pixels"],
                    "final": {k: {"fraction": f, "rel_error": e} for k, (f, e) in res["final"].items()},
                    "stopped": stop_file.exists()})
            except Exception as e:
                jobs[jid].update(status="error", error=str(e), trace=traceback.format_exc())
            jobs[jid]["seconds"] = round(time.time() - jobs[jid]["t0"], 2)

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    @app.post("/api/agent/stop")
    def agent_stop():
        stop_file.parent.mkdir(exist_ok=True)
        stop_file.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        return {"stopped": True}

    @app.get("/api/agent/{jid}")
    def agent_status(jid: str):
        if jid not in jobs:
            raise HTTPException(404)
        a = agents.get(jid)
        doc = {k: v for k, v in jobs[jid].items() if not k.startswith("_")}
        if a:
            ag, inst = a["agent"], a["inst"]
            doc["trace"] = json.loads(pd.DataFrame(list(ag.trace)).to_json(orient="records"))
            doc["decisions"] = json.loads(pd.DataFrame(list(ag.decisions)[-300:]).to_json(
                orient="records"))
            doc["ledger_h"] = {k: v / 3600 for k, v in getattr(inst, "ledger", {}).items()}
            doc["clock_h"] = inst.clock_s / 3600
            if hasattr(inst, "rec"):
                calls = pd.Series([c["call"] for c in inst.rec.calls.entries], dtype=object)
                doc["autoscript_calls"] = calls.value_counts().to_dict()
        return JSONResponse(json.loads(json.dumps(doc, default=float)))

    @app.get("/api/recipe.json")
    def recipe_json():
        import yaml
        p = proj.root / "studio_recipe.yaml"
        return yaml.safe_load(p.read_text()) if p.exists() else {"steps": []}

    @app.get("/api/recipe")
    def recipe():
        p = proj.root / "studio_recipe.yaml"
        return Response(p.read_text() if p.exists() else "# no steps yet\n",
                        media_type="text/yaml")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _log_recipe(root: Path, req: dict):
    """Every UI action is appended as a recipe step (reproducibility)."""
    import yaml
    p = root / "studio_recipe.yaml"
    doc = yaml.safe_load(p.read_text()) if p.exists() else {"name": root.name, "steps": []}
    doc["steps"].append({"op": req["op"], "inputs": req.get("inputs", {}),
                         "params": req.get("params", {})})
    p.write_text(yaml.safe_dump(doc, sort_keys=False))


def serve(project_dir: str | Path = ".", host: str = "127.0.0.1", port: int = 8765):
    import uvicorn
    print(f"NUScope-TOMO Studio → http://{host}:{port}   (project: {Path(project_dir).resolve()})")
    uvicorn.run(create_app(project_dir), host=host, port=port, log_level="warning")
