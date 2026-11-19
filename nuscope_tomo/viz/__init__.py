"""Visualisation: GPU renders, animations and figure panels.

The renderers need PyVista (VTK, about 0.5 GB), which only figure and animation
work uses. They load on first use, so the Studio and recipes run without it.
"""
from .palette import PALETTE  # noqa: F401

_RENDER = ("mask_surface", "phase_surfaces", "render_objects", "render_phases", "render_volume",
           "to_grid", "turntable", "render_network", "render_field")


def __getattr__(name):
    if name in _RENDER:
        try:
            from . import render
        except ImportError as e:
            raise ImportError(f"nuscope_tomo.viz.{name} needs PyVista for 3D rendering: pip install pyvista ({e})") from e
        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PALETTE", *_RENDER]
