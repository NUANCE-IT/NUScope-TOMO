"""Self-driving slice-and-view.

Per slice the agent: mills, images, checks focus (sharpness vs a running
reference → autofocus and re-image), measures drift against the previous
image (→ beam shift), keeps the SNR near a target by adapting dwell time,
segments the slice, adapts the next slice thickness to how fast the
microstructure is changing (within the advisor's bounds), takes sparse,
context-driven EDS every few slices, and stops as soon as the requested
precision on every phase fraction is reached — or a time/depth budget runs
out. Every decision is logged with its reason (an audit trail).

A ``conventional`` policy (fixed thickness, dwell and full EDS maps, periodic
autofocus) is provided for comparison on the digital twin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from skimage.restoration import estimate_sigma

from ..analyze.confidence import integral_range_cov
from .instrument import Instrument, SafetyError


@dataclass
class Policy:
    pixel_nm: float = 10.0
    slice_nm: float = 20.0
    min_slice_nm: float = 20.0
    max_slice_nm: float = 60.0
    adapt_thickness: bool = True
    change_target: float = 0.20        # fraction of pixels changing class per slice
    thickness_step_nm: float = 20.0
    adapt_after: int = 10              # slices at the base thickness before adapting
    control_every: int = 5             # minimum slices between dwell / thickness changes
    dwell_us: float = 3.0
    adapt_dwell: bool = True
    snr_target: float = 9.5
    min_dwell_us: float = 0.5
    max_dwell_us: float = 10.0
    autofocus_on_blur: bool = True
    sharp_drop: float = 0.8            # autofocus when sharpness < 80 % of running reference
    autofocus_every: int = 0           # periodic autofocus (conventional), 0 = off
    drift_correct: bool = True
    eds_every: int = 25                # slices between EDS acquisitions (0 = none)
    eds_sparse_budget: float = 0.05    # fraction of EDS pixels when sparse (1.0 = full map)
    eds_bin: int = 6                   # EDS pixel = eds_bin image pixels
    eds_dwell_ms: float = 5.0
    target_rel_error: float = 0.05
    precision_phases: Optional[list] = None  # names that must reach the target (None = all >1 %)
    stop_on_precision: bool = True
    stats_every: int = 25
    max_time_s: float = 48 * 3600.0
    max_depth_nm: float = 1e9


def _sharpness(img):
    a = ndi.gaussian_filter(img.astype(np.float32), 1.0)
    return float(ndi.laplace(a).var() / (a.var() + 1e-6))


def _snr(img):
    a = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(a, (2, 98))
    return float((hi - lo) / (estimate_sigma(a) + 1e-6))


class SliceAndViewAgent:
    def __init__(self, instrument: Instrument, classifier, policy: Optional[Policy] = None,
                 class_names: Optional[dict] = None, classifier_pixel_nm: float = 10.0,
                 log: Callable[[str], None] = lambda *_: None):
        self.inst, self.clf = instrument, classifier
        self.p = policy or Policy()
        self.names = class_names or getattr(classifier, "class_names", {})
        self.clf_px = classifier_pixel_nm
        self.log = log
        self.decisions: list[dict] = []
        self.trace: list[dict] = []
        self.labels: list[np.ndarray] = []
        self.eds_records: list[dict] = []

    def _decide(self, action, reason, **kw):
        self.decisions.append({"slice": self.inst.slices, "depth_um": self.inst.depth_nm / 1e3,
                               "t_h": self.inst.clock_s / 3600, "action": action,
                               "reason": reason, **kw})

    def _segment(self, img):
        from skimage.restoration import denoise_nl_means
        a = img.astype(np.float32) / 255
        sig = float(estimate_sigma(a))
        d = denoise_nl_means(a, h=0.8 * sig, sigma=sig, patch_size=5, patch_distance=6,
                             fast_mode=True) * 255
        if abs(self.p.pixel_nm - self.clf_px) > 1e-6:  # classifier was trained at another pixel
            d = ndi.zoom(d, self.p.pixel_nm / self.clf_px, order=1)
        lab = self.clf.predict_slice(np.clip(d, 0, 255).astype(np.uint8))
        if abs(self.p.pixel_nm - self.clf_px) > 1e-6:
            lab = lab[:: int(round(self.p.pixel_nm / self.clf_px)),
                      :: int(round(self.p.pixel_nm / self.clf_px))]
        return lab, d

    def _precision(self):
        """Running fractions and relative 95 % errors from the labels so far."""
        lab = np.stack([l[: self.labels[0].shape[0], : self.labels[0].shape[1]]
                        for l in self.labels])
        th = np.array([t["thickness_nm"] for t in self.trace])
        z_nm = float(th.mean())
        sp = (z_nm, self.p.pixel_nm, self.p.pixel_nm)
        f = max(1, int(round(z_nm / self.p.pixel_nm)))  # ~isotropic grid for the covariance
        small = lab[:, ::f, ::f]
        V = lab.size * np.prod(sp) / 1e9
        out = {}
        for c, n in self.names.items():
            c = int(c)
            m = lab == c
            p = float(m.mean())
            if p < 0.01:
                continue
            A3 = integral_range_cov(small == c, (z_nm, self.p.pixel_nm * f, self.p.pixel_nm * f))
            out[n] = (p, 2 * np.sqrt(p * (1 - p) * A3 / V) / p)
        return out

    def _eds(self, img, d):
        from ..sparse import active_sparse
        B = self.p.eds_bin
        ny, nx = img.shape[0] // B, img.shape[1] // B
        coords = np.argwhere(np.ones((ny, nx), bool))

        def measure(idx):
            return self.inst.eds(coords[idx] * B + B // 2, self.p.pixel_nm, self.p.eds_dwell_ms)

        if self.p.eds_sparse_budget >= 1.0:
            comp = measure(np.arange(len(coords))).reshape(ny, nx, -1)
            n = len(coords)
        else:
            r = active_sparse(d.astype(np.float32), (ny, nx), B, measure,
                              budget=self.p.eds_sparse_budget, init=self.p.eds_sparse_budget / 4,
                              rounds=3)
            comp, n = r["mean"], int(r["measured_mask"].sum())
        self.eds_records.append({"slice": self.inst.slices, "pixels": n, "of": len(coords),
                                 "comp": comp})
        self._decide("eds", f"{n} of {len(coords)} EDS pixels "
                     f"({'sparse, context-driven' if n < len(coords) else 'full map'})")

    def run(self) -> dict:
        p, inst = self.p, self.inst
        prev, sharp_ref = None, []
        th = p.slice_nm
        reached = False
        while True:
            if inst.stop_requested():
                self._decide("stop", "operator stop file found")
                break
            if inst.clock_s > p.max_time_s or inst.depth_nm + th > p.max_depth_nm:
                self._decide("stop", "time or depth budget reached")
                break
            try:
                inst.mill(th)
                if p.autofocus_every and inst.slices % p.autofocus_every == 0:
                    inst.autofocus()
                    self._decide("autofocus", f"scheduled every {p.autofocus_every} slices")
                img = inst.image(p.pixel_nm, p.dwell_us)
            except SafetyError as e:
                self._decide("stop", f"safety: {e}")
                break
            sh = _sharpness(img)
            if p.autofocus_on_blur and len(sharp_ref) >= 5 and sh < p.sharp_drop * np.median(sharp_ref[-15:]):
                inst.autofocus()
                img = inst.image(p.pixel_nm, p.dwell_us)
                self._decide("autofocus", f"sharpness {sh:.4f} < {p.sharp_drop:.0%} of reference")
                sh = _sharpness(img)
            sharp_ref.append(sh)
            if prev is not None and p.drift_correct:
                s, _, _ = phase_cross_correlation(prev.astype(np.float32), img.astype(np.float32),
                                                  upsample_factor=4, normalization=None)
                if np.hypot(*s) > 0.75:
                    inst.beam_shift(-s[0] * p.pixel_nm, -s[1] * p.pixel_nm)
                    img = np.clip(ndi.shift(img.astype(np.float32), s, order=1, mode="nearest"),
                                  0, 255).astype(np.uint8)
                    self._decide("beam_shift", f"drift {np.hypot(*s) * p.pixel_nm:.0f} nm",
                                 dy_nm=float(s[0] * p.pixel_nm), dx_nm=float(s[1] * p.pixel_nm))
            snr = _snr(img)
            n_done = len(self.trace) + 1
            can_act = n_done >= p.adapt_after and n_done - getattr(self, "_last_ctl", -99) >= p.control_every
            if p.adapt_dwell and can_act and len(self.trace) >= 4:
                snr_med = float(np.median([t["snr"] for t in self.trace[-4:]] + [snr]))
                nd = p.dwell_us
                if snr_med < 0.9 * p.snr_target:
                    nd = min(p.max_dwell_us, p.dwell_us * 1.3)
                elif snr_med > 1.25 * p.snr_target:
                    nd = max(p.min_dwell_us, p.dwell_us / 1.2)
                if nd != p.dwell_us:
                    self._decide("dwell", f"median SNR {snr_med:.1f} vs target {p.snr_target}",
                                 dwell_from=p.dwell_us, dwell_to=nd)
                    p.dwell_us = nd
                    self._last_ctl = n_done
            lab, d = self._segment(img)
            change = float((lab != self.labels[-1][: lab.shape[0], : lab.shape[1]]).mean()) \
                if self.labels else np.nan
            self.labels.append(lab)
            self.trace.append({"slice": inst.slices, "depth_um": inst.depth_nm / 1e3,
                               "t_h": inst.clock_s / 3600, "thickness_nm": th,
                               "dwell_us": p.dwell_us, "snr": snr, "sharpness": sh,
                               "change": change})
            n_done = len(self.trace)
            can_act = n_done >= p.adapt_after and \
                n_done - getattr(self, "_last_th", -99) >= p.control_every
            if p.adapt_thickness and np.isfinite(change) and can_act:
                change = float(np.nanmedian([t["change"] for t in self.trace[-5:]]))
                # thicker slices while the structure changes slowly, thinner when it changes fast
                step = p.thickness_step_nm
                new = th
                if change < 0.66 * p.change_target:
                    new = th + step
                elif change > 1.3 * p.change_target:
                    new = th - step
                new = float(np.clip(new, p.min_slice_nm, p.max_slice_nm))
                if new != th:
                    self._last_th = n_done
                    self._decide("thickness", f"{change:.0%} of pixels changed class "
                                 f"(target {p.change_target:.0%})", from_nm=th, to_nm=new)
                    th = new
            if p.eds_every and inst.slices % p.eds_every == 0:
                self._eds(img, d)
            prev = img
            if p.stats_every and len(self.labels) % p.stats_every == 0 and len(self.labels) >= 50:
                pr = self._precision()
                self.trace[-1].update({f"err_{n}": e for n, (_, e) in pr.items()})
                self.trace[-1].update({f"frac_{n}": f for n, (f, _) in pr.items()})
                need = {n: v for n, v in pr.items()
                        if p.precision_phases is None or n in p.precision_phases}
                if p.stop_on_precision and need and all(e <= p.target_rel_error
                                                        for _, e in need.values()):
                    self._decide("stop", f"precision ±{p.target_rel_error:.0%} reached on all phases")
                    reached = True
                    break
        final = self._precision() if len(self.labels) >= 10 else {}
        return {"trace": pd.DataFrame(self.trace), "decisions": pd.DataFrame(self.decisions),
                "final": final, "reached": reached, "policy": asdict(p),
                "time_h": inst.clock_s / 3600, "depth_um": inst.depth_nm / 1e3,
                "slices": inst.slices,
                "eds_pixels": int(sum(r["pixels"] for r in self.eds_records)),
                "eds_pixels_full": int(sum(r["of"] for r in self.eds_records)),
                "ledger_h": {k: v / 3600 for k, v in
                                                    getattr(inst, "ledger", {}).items()}}


def conventional_policy(**kw) -> Policy:
    """What a fixed Auto Slice & View recipe does: no adaptation, full EDS maps."""
    base = dict(adapt_thickness=False, adapt_dwell=False, autofocus_on_blur=False,
                autofocus_every=50, drift_correct=True, eds_sparse_budget=1.0,
                stop_on_precision=False)
    base.update(kw)
    return Policy(**base)


__all__ = ["Policy", "SliceAndViewAgent", "conventional_policy"]
