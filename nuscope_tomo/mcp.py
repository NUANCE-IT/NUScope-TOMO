"""NUScope-TOMO inside NUANCE-MCP: tomography projects as tools for the LLM.

NUANCE-MCP (https://github.com/NUANCE-IT/NUANCE-MCP) loads this module through
the ``nuance_mcp.plugins`` entry point. The tools let a model inspect a
project, read its results and run a recipe, without touching the microscope:

* ``tomo_project``   volumes (shape, voxel size in nm, phase names), tables, figures
* ``tomo_table``     a result table (phase fractions, particles, quality ...), summarized
* ``tomo_volume``    shape, voxel size and metadata of one volume
* ``tomo_ops``       the recipe operations and their parameters
* ``tomo_phase_fractions`` volume fraction and surface density of each phase of a labelled
  volume, with sampling standard errors from the covariance integral range
* ``tomo_run``       run a YAML recipe in the background; ``tomo_job`` follows it

Recipe runs write their outputs and provenance into the project folder, as
``nuscope-tomo run`` does.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from . import __version__

_JOBS: dict[str, dict] = {}


def _volumes(root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(root.rglob("*.zarr")):
        if (p / "zarr.json").exists() and not p.name.startswith("_") \
                and not any(q.suffix == ".zarr" for q in p.parents):
            out[str(p.relative_to(root).with_suffix(""))] = p
    return out


def _inside(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root.resolve() not in (p, *p.parents):
        raise ValueError(f"{rel} is outside the project")
    return p


def register(ctx) -> None:
    """Entry point for NUANCE-MCP (``nuance_mcp.plugins``)."""
    mcp = ctx.mcp

    @mcp.tool(name="tomo_project")
    def tomo_project(project: str) -> dict:
        """Summarize a NUScope-TOMO project folder: its volumes (shape, voxel size in nm, phase names),
        result tables and figures."""
        from .core.volume import Volume
        root = Path(project).expanduser().resolve()
        if not root.is_dir():
            return {"error": f"no folder {root}"}
        vols = []
        for vid, p in _volumes(root).items():
            try:
                v = Volume.from_zarr(p)
                vols.append({"id": vid, "shape": list(v.shape), "dtype": str(v.dtype), "kind": v.kind,
                             "voxel_nm": list(v.voxel_size.as_tuple()), "classes": v.meta.get("classes", {})})
            except Exception as e:
                vols.append({"id": vid, "error": str(e)})
        return {"project": str(root), "nuscope_tomo": __version__, "volumes": vols,
                "tables": sorted(str(p.relative_to(root)) for p in root.rglob("*.csv")),
                "figures": sorted(str(p.relative_to(root)) for ext in ("png", "mp4")
                                  for p in root.rglob(f"*.{ext}"))[:100]}

    @mcp.tool(name="tomo_table")
    def tomo_table(project: str, table: str, limit: int = 20) -> dict:
        """Read a result table of a project (path relative to the project, e.g. tables/phases.csv):
        columns, the first rows and summary statistics."""
        import pandas as pd
        df = pd.read_csv(_inside(Path(project).expanduser(), table))
        return {"table": table, "rows": len(df), "columns": list(df.columns),
                "head": df.head(max(1, min(limit, 200))).to_dict(orient="records"),
                "describe": df.describe().round(4).to_dict() if len(df) else {}}

    @mcp.tool(name="tomo_volume")
    def tomo_volume(path: str) -> dict:
        """Shape, voxel size (nm), physical size (µm) and metadata of a volume (.zarr or a TIFF folder)."""
        from .core.volume import Volume
        from .io.stack import read_tiff_folder
        p = Path(path).expanduser()
        v = Volume.from_zarr(p) if p.suffix == ".zarr" else read_tiff_folder(p)
        meta = {k: val for k, val in v.meta.items() if not isinstance(val, (list, dict)) or len(val) < 20}
        return {"name": v.name, "shape": list(v.shape), "dtype": str(v.dtype),
                "voxel_nm": list(v.voxel_size.as_tuple()),
                "size_um": [round(s / 1e3, 3) for s in v.extent_nm], "meta": meta}

    @mcp.tool(name="tomo_phase_fractions")
    def tomo_phase_fractions(labels: str) -> dict:
        """Volume fraction, ±sampling standard error and surface density (1/µm) of every phase in a
        labelled volume (.zarr written by a recipe, e.g. volumes/phases.zarr). The error says how far the
        fraction could move with a different field of view of the same size."""
        import math
        from .cohort import summarize
        from .core.volume import Volume
        vol = Volume.from_zarr(Path(labels).expanduser())
        df = summarize(vol)["phases"]
        rows = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
                for r in df.to_dict(orient="records")]
        return {"labels": str(labels), "voxel_nm": list(vol.voxel_size.as_tuple()),
                "volume_um3": rows[0]["volume_um3"] if rows else 0.0, "phases": rows}

    @mcp.tool(name="tomo_ops")
    def tomo_ops() -> dict:
        """The operations a NUScope-TOMO recipe can use, with their default parameters."""
        from .core.registry import list_ops
        return {name: {"summary": spec.summary, "params": {k: repr(d) for k, d in spec.params().items()}}
                for name, spec in list_ops().items()}

    @mcp.tool(name="tomo_run")
    def tomo_run(recipe: str, project: Optional[str] = None) -> dict:
        """Run a YAML recipe in the background (alignment, correction, segmentation, measurement ...).
        Returns a job id; follow it with tomo_job. Outputs are written into the project folder."""
        from .core.recipe import run_recipe
        jid = uuid.uuid4().hex[:8]
        _JOBS[jid] = {"id": jid, "recipe": recipe, "project": project, "status": "running", "t0": time.time()}

        def work():
            try:
                pipe = run_recipe(recipe, project)
                _JOBS[jid].update(status="done", project=str(pipe.project))
            except Exception as e:
                _JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}",
                                  trace=traceback.format_exc()[-2000:])
            _JOBS[jid]["seconds"] = round(time.time() - _JOBS[jid]["t0"], 1)

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid, "status": "running"}

    @mcp.tool(name="tomo_job")
    def tomo_job(job: str) -> dict:
        """Status of a recipe run started with tomo_run."""
        j = _JOBS.get(job)
        if j is None:
            return {"error": f"no job {job}", "jobs": sorted(_JOBS)}
        return {k: v for k, v in j.items() if k != "t0"} | {"elapsed_s": round(time.time() - j["t0"], 1)}


__all__ = ["register"]
