"""Shared colormap list + 8-bit LUT cache, used by both the 3D view (which
hands cmap names straight to PyVista/matplotlib) and the 2D slice view
(which needs an actual RGB lookup table to color a QImage by hand).
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Display name -> matplotlib colormap name. Grouped: perceptually-uniform
# sequential first (best default for real intensity data), then classic/
# diverging, then a qualitative one for label/segmentation volumes where
# adjacent integers are different objects, not a gradient.
COLORMAPS: dict[str, str] = {
    "Grayscale": "gray",
    "Bone": "bone",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Magma": "magma",
    "Inferno": "inferno",
    "Cividis": "cividis",
    "Turbo": "turbo",
    "Hot": "hot",
    "Coolwarm": "coolwarm",
    "Labels (qualitative)": "tab20",
}

DEFAULT_INTENSITY_COLORMAP = "Bone"
DEFAULT_LABEL_COLORMAP = "Labels (qualitative)"

_lut_cache: dict[str, np.ndarray] = {}


def lut_for(display_name: str) -> np.ndarray:
    """256x3 uint8 RGB lookup table for a colormap, by its display name."""
    if display_name not in _lut_cache:
        cmap_name = COLORMAPS.get(display_name, display_name)
        cmap = mpl.colormaps[cmap_name].resampled(256)
        _lut_cache[display_name] = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _lut_cache[display_name]


def mpl_name(display_name: str) -> str:
    """The underlying matplotlib/PyVista colormap name for a display name."""
    return COLORMAPS.get(display_name, display_name)
