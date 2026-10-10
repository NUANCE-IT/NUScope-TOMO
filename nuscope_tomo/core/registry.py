"""Operation registry: every processing function that can appear in a recipe
(or behind a Studio button) registers itself here under a dotted name.

An operation takes its inputs as keyword arguments (Volumes, tables, paths)
plus plain JSON-able parameters, and returns a Volume, a pandas DataFrame, a
dict of named results, or None.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

_OPS: dict[str, "OpSpec"] = {}


@dataclass
class OpSpec:
    name: str
    func: Callable
    category: str
    summary: str
    inputs: tuple[str, ...]

    def params(self) -> dict:
        """Tunable parameters and their defaults (for UI forms / docs)."""
        out = {}
        for p in inspect.signature(self.func).parameters.values():
            if p.name in self.inputs or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.name == "progress":
                continue
            out[p.name] = None if p.default is inspect._empty else p.default
        return out


def op(name: str, inputs: tuple[str, ...] = ("volume",)):
    """Decorator registering ``func`` as recipe operation ``name``."""

    def wrap(func):
        doc = (func.__doc__ or "").strip().splitlines()
        _OPS[name] = OpSpec(name, func, name.split(".")[0], doc[0] if doc else "", inputs)
        return func

    return wrap


def get_op(name: str) -> OpSpec:
    _ensure_loaded()
    if name not in _OPS:
        raise KeyError(f"Unknown operation {name!r}. Known: {sorted(_OPS)}")
    return _OPS[name]


def list_ops() -> dict[str, OpSpec]:
    _ensure_loaded()
    return dict(sorted(_OPS.items()))


def _ensure_loaded():
    # Importing the subpackages registers their operations.
    from .. import advisor, analyze, chem, io, preprocess, segment, viz  # noqa: F401
    try:  # deep learning is optional (needs torch)
        from .. import ai  # noqa: F401
    except ImportError:
        pass
