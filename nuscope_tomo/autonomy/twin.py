"""Digital twin of a FIB-SEM slice-and-view run, built from a measured stack.

The "true" specimen is a clean (denoised) volume plus its phase map. The
twin reproduces what an operator fights during acquisition:

* milling: the face advances by the requested thickness (true volume
  sampled at the nearest available plane); milling time ∝ thickness;
* imaging: larger pixels average the signal, shot noise falls as
  1/√dwell, imaging time = pixels × dwell + overhead;
* focus drifts slowly (blur grows with slices since the last autofocus);
* the image drifts (random walk) until beam shift compensates it;
* EDS: composition per phase with an interaction blur and Poisson counts
  ∝ dwell (forward model of :mod:`nuscope_tomo.chem.simulate`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from ..core.volume import Volume
from .instrument import Instrument, SafetyLimits


class DigitalTwin(Instrument):
    def __init__(self, clean: Volume, phases: Volume, compositions: dict,
                 limits: Optional[SafetyLimits] = None, noise_at_1us: float = 18.0,
                 mill_s_per_nm: float = 1.2, image_overhead_s: float = 2.0,
                 autofocus_s: float = 15.0, defocus_px_per_slice: float = 0.02,
                 drift_nm_per_slice: float = 3.0, eds_overhead_s: float = 5.0,
                 eds_counts_per_ms: float = 8.0, slice_overhead_s: float = 8.0, seed: int = 0):
        self.clean, self.phases = clean.numpy().astype(np.float32), phases.numpy()
        self.vs = clean.voxel_size
        self.comp = compositions
        self.elements = sorted({e for c in compositions.values() for e in c})
        self.limits = limits or SafetyLimits()
        self.noise1 = noise_at_1us
        self.mill_rate, self.img_over, self.af_s = mill_s_per_nm, image_overhead_s, autofocus_s
        self.defocus_rate, self.drift_step = defocus_px_per_slice, drift_nm_per_slice
        self.eds_over, self.eds_rate = eds_overhead_s, eds_counts_per_ms
        self.slice_over = slice_overhead_s  # beam switching, pattern setup, image transfer
        self.rng = np.random.default_rng(seed)
        self.clock_s = self.depth_nm = 0.0
        self.slices = 0
        self.since_af = 0
        self.drift = np.zeros(2)   # nm, accumulated image offset
        self.ledger = {"mill": 0.0, "image": 0.0, "autofocus": 0.0, "eds": 0.0, "overhead": 0.0}

    # ---------------------------------------------------------------- helpers
    @property
    def z_index(self) -> int:
        return int(min(round(self.depth_nm / self.vs.z), self.clean.shape[0] - 1))

    def _spend(self, kind: str, s: float):
        self.clock_s += s
        self.ledger[kind] += s

    def truth_slice(self) -> np.ndarray:
        return self.phases[self.z_index]

    # ---------------------------------------------------------------- actions
    def mill(self, thickness_nm: float) -> None:
        self.check(thickness_nm=thickness_nm)
        if self.depth_nm + thickness_nm > (self.clean.shape[0] - 1) * self.vs.z:
            from .instrument import SafetyError
            raise SafetyError("end of specimen (twin volume) reached")
        self.depth_nm += thickness_nm
        self.slices += 1
        self.since_af += 1
        self.drift += self.rng.normal(0, self.drift_step, 2)
        self._spend("mill", thickness_nm * self.mill_rate)
        self._spend("overhead", self.slice_over)

    def image(self, pixel_nm: float, dwell_us: float) -> np.ndarray:
        self.check(dwell_us=dwell_us)
        img = self.clean[self.z_index]
        img = ndi.shift(img, self.drift / self.vs.y, order=1, mode="nearest")
        blur = self.defocus_rate * self.since_af
        if blur > 0.05:
            img = ndi.gaussian_filter(img, blur)
        f = max(1, int(round(pixel_nm / self.vs.y)))
        h, w = img.shape[0] // f * f, img.shape[1] // f * f
        img = img[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))
        sd = self.noise1 / np.sqrt(dwell_us * f * f)  # bigger pixels collect more signal
        out = np.clip(img + self.rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)
        self._spend("image", out.size * dwell_us * 1e-6 + self.img_over)
        return out

    def autofocus(self) -> None:
        self.since_af = 0
        self._spend("autofocus", self.af_s)

    def auto_contrast(self) -> None:
        self._spend("autofocus", 3.0)

    def beam_shift(self, dy_nm: float, dx_nm: float) -> None:
        self.drift -= np.array([dy_nm, dx_nm])

    def eds(self, points_yx: np.ndarray, pixel_nm: float, dwell_ms: float) -> np.ndarray:
        """Measured composition (N, E) at points given in image-pixel coordinates."""
        ph = self.phases[self.z_index].astype(int)
        f = pixel_nm / self.vs.y
        maps = []
        for e in self.elements:
            lut = np.zeros(ph.max() + 1, np.float32)
            for c, comp in self.comp.items():
                if c <= ph.max():
                    lut[c] = comp.get(e, 0.0)
            maps.append(ndi.gaussian_filter(lut[ph], 60 / self.vs.y))
        yy = np.clip((points_yx[:, 0] + 0.5) * f, 0, ph.shape[0] - 1).astype(int)
        xx = np.clip((points_yx[:, 1] + 0.5) * f, 0, ph.shape[1] - 1).astype(int)
        lam = np.stack([m[yy, xx] for m in maps], -1) * self.eds_rate * dwell_ms
        cnt = self.rng.poisson(lam)
        self._spend("eds", len(points_yx) * dwell_ms * 1e-3 + self.eds_over)
        return cnt / np.maximum(cnt.sum(-1, keepdims=True), 1)
