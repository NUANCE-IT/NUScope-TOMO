"""Forward model for EDS slice-and-view maps, used for validation and demos.

Given a phase map on the SE grid and a composition per phase, generates what
an EDS detector would record: coarser pixels, an X-ray interaction-volume
blur, a field-of-view offset relative to SE, and Poisson counting noise.
Because the truth is known, registration and phase recovery can be scored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from ..core.volume import VoxelSize
from .eds import EDSStack


def simulate_eds(phases: np.ndarray, compositions: dict[int, dict[str, float]],
                 se_voxel: VoxelSize, bin_xy: int = 8, every: int = 4,
                 counts_per_pixel: float = 60.0, psf_nm: float = 80.0,
                 shift_px: tuple[float, float] = (1.3, -0.7), seed: int = 0,
                 write_to: Optional[str | Path] = None) -> tuple[EDSStack, dict]:
    rng = np.random.default_rng(seed)
    elements = sorted({e for c in compositions.values() for e in c})
    z_index = np.arange(0, phases.shape[0], every)
    ny, nx = phases.shape[1] // bin_xy, phases.shape[2] // bin_xy
    maps = {e: np.zeros((len(z_index), ny, nx), np.float32) for e in elements}
    psf = psf_nm / se_voxel.y
    for k, z in enumerate(z_index):
        ph = phases[z]
        for e in elements:
            lut = np.zeros(int(ph.max()) + 1, np.float32)
            for c, comp in compositions.items():
                if c <= ph.max():
                    lut[c] = comp.get(e, 0.0)
            img = ndi.gaussian_filter(lut[ph], psf)
            lo = img[: ny * bin_xy, : nx * bin_xy].reshape(ny, bin_xy, nx, bin_xy).mean((1, 3))
            lo = ndi.shift(lo, shift_px, order=1, mode="nearest")
            maps[e][k] = rng.poisson(lo * counts_per_pixel).astype(np.float32)
    vs = VoxelSize(se_voxel.z * every, se_voxel.y * bin_xy, se_voxel.x * bin_xy)
    eds = EDSStack(maps, vs, z_index)
    truth = {"shift_px": shift_px, "bin_xy": bin_xy, "every": every, "elements": elements}
    if write_to:
        import tifffile
        root = Path(write_to)
        for e, m in maps.items():
            (root / e).mkdir(parents=True, exist_ok=True)
            for k in range(m.shape[0]):
                tifffile.imwrite(root / e / f"{e}_{k:04d}.tif", m[k].astype(np.uint16))
    return eds, truth
