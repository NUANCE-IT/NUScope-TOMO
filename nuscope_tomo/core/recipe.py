"""Recipes: reproducible YAML pipelines (the scriptable counterpart of Avizo
recipes). Every executed step is logged with its parameters, timing and
software versions in ``<project>/provenance.json``.

Example
-------
.. code-block:: yaml

    name: battery-cathode
    project: projects/cathode
    input:
      tiff_folder: /data/slice_and_view
      voxel_size_nm: [20, 5, 5]
    steps:
      - op: preprocess.auto_crop
      - op: preprocess.align
        save: true
      - id: seg
        op: segment.multi_otsu
        params: {classes: 3}
      - op: analyze.phase_metrics
        inputs: {labels: seg}

A step's single input defaults to the previous step's Volume output.
``inputs`` maps parameter names to earlier step ids for multi-input ops.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yaml

from .. import __version__
from .registry import get_op
from .volume import Volume, VoxelSize, _jsonable


@dataclass
class Step:
    op: str
    id: Optional[str] = None
    params: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    save: bool = False


@dataclass
class Recipe:
    name: str
    steps: list[Step]
    input: dict = field(default_factory=dict)
    project: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        steps = [Step(**s) for s in raw.get("steps", [])]
        rec = cls(raw.get("name", path.stem), steps, raw.get("input", {}), raw.get("project"))
        rec._base = path.parent
        return rec

    def to_yaml(self) -> str:
        return yaml.safe_dump({
            "name": self.name, "project": self.project, "input": self.input,
            "steps": [{k: v for k, v in s.__dict__.items() if v not in (None, {}, False)}
                      for s in self.steps]}, sort_keys=False)


class Pipeline:
    """Executes a recipe, keeping every step output addressable by id."""

    def __init__(self, project_dir: str | Path,
                 log: Callable[[str], None] = print):
        self.project = Path(project_dir)
        self.project.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}
        self.provenance: list[dict] = []
        self.log = log

    # ---------------------------------------------------------------- inputs
    def load_input(self, spec: dict, base: Path = Path(".")) -> Volume:
        from ..io.stack import read_tiff_folder, read_tiff_stack

        vs = spec.get("voxel_size_nm")
        voxel = VoxelSize(*vs) if vs else None
        if "zarr" in spec:
            vol = Volume.from_zarr(_resolve(spec["zarr"], base))
            if voxel:
                vol.voxel_size = voxel
        elif "tiff_folder" in spec:
            vol = read_tiff_folder(_resolve(spec["tiff_folder"], base), voxel_size=voxel,
                                   cache=self.project / "raw.zarr")
        elif "tiff_stack" in spec:
            vol = read_tiff_stack(_resolve(spec["tiff_stack"], base), voxel_size=voxel)
        else:
            raise ValueError(f"Unsupported input spec {spec}")
        self.results["input"] = vol
        self._last_volume = "input"
        self.log(f"input  {vol}")
        return vol

    # ------------------------------------------------------------------- run
    def run_step(self, step: Step) -> Any:
        spec = get_op(step.op)
        sid = step.id or f"{len(self.provenance):02d}_{step.op.split('.')[-1]}"
        kwargs = dict(step.params)
        explicit = dict(step.inputs)
        for name in spec.inputs:
            if name in explicit:
                kwargs[name] = self._lookup(explicit[name])
            elif name == spec.inputs[0]:
                kwargs[name] = self.results[self._last_volume]
        t0 = time.perf_counter()
        out = spec.func(**kwargs)
        self.results[sid] = out
        if isinstance(out, Volume):
            self._last_volume = sid
            if step.save:
                out = out.to_zarr(self.project / "volumes" / f"{sid}.zarr")
                self.results[sid] = out
        elif isinstance(out, pd.DataFrame):
            (self.project / "tables").mkdir(exist_ok=True)
            out.to_csv(self.project / "tables" / f"{sid}.csv", index=False)
        elif isinstance(out, dict):
            for k, v in out.items():
                self.results[f"{sid}.{k}"] = v
                if isinstance(v, pd.DataFrame):
                    (self.project / "tables").mkdir(exist_ok=True)
                    v.to_csv(self.project / "tables" / f"{sid}.{k}.csv", index=False)
            if "volume" in out and isinstance(out["volume"], Volume):
                vol = out["volume"]
                if step.save:
                    vol = vol.to_zarr(self.project / "volumes" / f"{sid}.zarr")
                self.results[sid] = self.results[f"{sid}.volume"] = vol
                self._last_volume = sid
        dt = time.perf_counter() - t0
        self.provenance.append({
            "id": sid, "op": step.op, "params": _jsonable(step.params),
            "inputs": explicit, "seconds": round(dt, 3), "output": _describe(out)})
        self.log(f"{sid:<28s} {dt:7.1f}s  {_describe(out)}")
        return out

    def _lookup(self, key: str):
        if key not in self.results:
            raise KeyError(f"No result {key!r}; available: {list(self.results)}")
        return self.results[key]

    def run(self, recipe: Recipe) -> dict[str, Any]:
        base = getattr(recipe, "_base", Path("."))
        self.load_input(recipe.input, base)
        for step in recipe.steps:
            self.run_step(step)
        self.write_provenance(recipe)
        return self.results

    def write_provenance(self, recipe: Optional[Recipe] = None):
        import dask, numpy, scipy, skimage, sklearn  # noqa: E401

        doc = {
            "nuscope_tomo": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {m.__name__: m.__version__ for m in (numpy, scipy, skimage, sklearn, dask)},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recipe": recipe.to_yaml() if recipe else None,
            "steps": self.provenance,
        }
        (self.project / "provenance.json").write_text(json.dumps(doc, indent=2))


def run_recipe(path: str | Path, project: Optional[str | Path] = None, **kw) -> Pipeline:
    recipe = Recipe.load(path)
    proj = Path(project or recipe.project or Path("projects") / recipe.name)
    pipe = Pipeline(proj, **kw)
    pipe.run(recipe)
    return pipe


def _resolve(p: str, base: Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() or p.exists() else base / p


def _describe(out) -> str:
    if isinstance(out, Volume):
        return f"Volume {out.shape} {out.dtype} [{out.kind}]"
    if isinstance(out, pd.DataFrame):
        return f"Table {out.shape[0]} rows x {out.shape[1]} cols"
    if isinstance(out, dict):
        return "{" + ", ".join(f"{k}: {_describe(v)}" for k, v in out.items()) + "}"
    if out is None:
        return "-"
    if isinstance(out, (int, float, str)):
        return repr(out)
    return type(out).__name__
