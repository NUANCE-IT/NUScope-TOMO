"""Intensity-based segmentation (out-of-core, elementwise after a sampled
histogram): manual band, Otsu, multi-Otsu, hysteresis."""
from __future__ import annotations

from typing import Optional, Sequence

import dask.array as da
import numpy as np
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu, threshold_otsu

from ..core.registry import op
from ..core.volume import Volume


def _sample(volume: Volume, step: int = 4) -> np.ndarray:
    return np.asarray(volume.data[::step, ::step, ::step]).ravel()


@op("segment.threshold")
def threshold(volume: Volume, low: Optional[float] = None, high: Optional[float] = None) -> Volume:
    """Binary mask low <= I <= high (Otsu if no bounds given)."""
    if low is None and high is None:
        low = float(threshold_otsu(_sample(volume)))
    lo = -np.inf if low is None else low
    hi = np.inf if high is None else high
    mask = ((volume.data >= lo) & (volume.data <= hi)).astype(np.uint8)
    return volume.derive(mask, name=f"{volume.name}_mask", kind="labels",
                         threshold=[low, high], classes={"1": "foreground"})


@op("segment.multi_otsu")
def multi_otsu(volume: Volume, classes: int = 3, names: Optional[Sequence[str]] = None) -> Volume:
    """Multi-class Otsu: labels 1..classes ordered dark → bright."""
    s = _sample(volume)
    s = s[s > 0] if (s > 0).mean() > 0.5 else s  # ignore alignment padding
    t = threshold_multiotsu(s, classes=classes)
    lab = (da.digitize(volume.data, t) + 1).astype(np.uint8)
    names = list(names) if names else [f"class_{i}" for i in range(1, classes + 1)]
    return volume.derive(lab, name=f"{volume.name}_phases", kind="labels",
                         thresholds=[float(v) for v in t],
                         classes={str(i + 1): n for i, n in enumerate(names)})


@op("segment.hysteresis")
def hysteresis(volume: Volume, low: float = 0.0, high: float = 0.0) -> Volume:
    """Hysteresis threshold (seeds above ``high`` grown through ``low``) — in memory."""
    m = apply_hysteresis_threshold(volume.numpy(), low, high).astype(np.uint8)
    return volume.derive(m, kind="labels", classes={"1": "foreground"})


__all__ = ["threshold", "multi_otsu", "hysteresis"]
