"""Instrument interface for autonomous slice-and-view, plus the Thermo Fisher
AutoScript backend.

The agent only talks to :class:`Instrument`. Two backends implement it:

* :class:`AutoScriptFIBSEM` — a Thermo Fisher DualBeam through AutoScript
  (``autoscript_sdb_microscope_client``). Calls follow the API as used by the
  open-source OpenFIBSEM package (DeMarco lab): ``SdbMicroscopeClient.connect``,
  ``imaging.grab_frame(GrabFrameSettings(...))``,
  ``beams.electron_beam.horizontal_field_width``, ``beam_shift``,
  ``auto_functions.run_auto_focus / run_auto_cb``,
  ``patterning.create_cleaning_cross_section(...)`` and ``patterning.run()``.
  **Not yet exercised on hardware.** It defaults to ``dry_run=True``, which
  records every call instead of sending it, and it refuses to act until
  :meth:`arm` is called with explicit operator confirmation.
* :class:`~nuscope_tomo.autonomy.twin.DigitalTwin` — a simulator built from
  a measured stack, used to develop and test policies without beam time.

EDS is not part of AutoScript (it is driven through the EDS vendor's API,
e.g. Oxford AZtec or EDAX); backends accept an ``eds_provider`` callable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class SafetyLimits:
    max_slices: int = 2000
    max_depth_nm: float = 50_000.0
    max_runtime_s: float = 24 * 3600.0
    min_slice_nm: float = 5.0
    max_slice_nm: float = 200.0
    min_dwell_us: float = 0.1
    max_dwell_us: float = 20.0
    stop_file: Optional[Path] = None  # create this file to stop after the current slice


@dataclass
class ActionLog:
    entries: list = field(default_factory=list)

    def add(self, **kw):
        kw.setdefault("t_wall", time.time())
        self.entries.append(kw)


class SafetyError(RuntimeError):
    pass


class Instrument:
    """What an autonomous slice-and-view agent needs from a microscope."""

    limits: SafetyLimits
    clock_s: float = 0.0       # instrument time spent (simulated or measured)
    depth_nm: float = 0.0
    slices: int = 0

    def mill(self, thickness_nm: float) -> None: ...
    def image(self, pixel_nm: float, dwell_us: float) -> np.ndarray: ...
    def autofocus(self) -> None: ...
    def auto_contrast(self) -> None: ...
    def beam_shift(self, dy_nm: float, dx_nm: float) -> None: ...
    def eds(self, points_yx: np.ndarray, pixel_nm: float, dwell_ms: float) -> np.ndarray: ...

    # shared guard rails --------------------------------------------------
    def check(self, thickness_nm: Optional[float] = None, dwell_us: Optional[float] = None):
        L = self.limits
        if self.slices >= L.max_slices:
            raise SafetyError("maximum number of slices reached")
        if thickness_nm is not None:
            if not L.min_slice_nm <= thickness_nm <= L.max_slice_nm:
                raise SafetyError(f"slice thickness {thickness_nm} nm outside limits")
            if self.depth_nm + thickness_nm > L.max_depth_nm:
                raise SafetyError("maximum milling depth reached")
        if dwell_us is not None and not L.min_dwell_us <= dwell_us <= L.max_dwell_us:
            raise SafetyError(f"dwell {dwell_us} µs outside limits")
        if self.clock_s > L.max_runtime_s:
            raise SafetyError("maximum run time reached")

    def stop_requested(self) -> bool:
        f = self.limits.stop_file
        return bool(f and Path(f).exists())


class AutoScriptFIBSEM(Instrument):
    """Thermo Fisher DualBeam via AutoScript (dry-run by default).

    Geometry: the cleaning cross-section that exposes each new face is a box
    of ``trench_width_um`` × thickness × ``mill_depth_um`` whose centre steps
    by the slice thickness along ``advance_axis`` ("y" by default) in ion-beam
    image coordinates, starting from ``face_center_um``.
    """

    ELECTRON, ION = 1, 2  # imaging view / device indices used by AutoScript

    def __init__(self, host: str = "localhost", port: Optional[int] = None,
                 dry_run: bool = True, limits: Optional[SafetyLimits] = None,
                 resolution: tuple[int, int] = (1536, 1024),
                 face_center_um: tuple[float, float] = (0.0, 0.0),
                 trench_width_um: float = 20.0, mill_depth_um: float = 10.0,
                 advance_axis: str = "y", advance_sign: float = -1.0,
                 milling_current_a: Optional[float] = None,
                 eds_provider: Optional[Callable] = None):
        self.limits = limits or SafetyLimits()
        self.dry_run, self.resolution = dry_run, resolution
        self.face = list(face_center_um)
        self.trench_w, self.mill_d = trench_width_um, mill_depth_um
        self.axis, self.sign = advance_axis, advance_sign
        self.current = milling_current_a
        self.eds_provider = eds_provider
        self.calls = ActionLog()
        self.armed = False
        self.shift = np.zeros(2)
        self.m = None
        if not dry_run:
            from autoscript_sdb_microscope_client import SdbMicroscopeClient
            self.m = SdbMicroscopeClient()
            self.m.connect(host) if port is None else self.m.connect(host=host, port=port)

    # -- safety ---------------------------------------------------------------
    def arm(self, confirmation: str):
        """Operator must type the exact phrase before the beam is used."""
        if confirmation != "I am at the microscope and accept autonomous milling":
            raise SafetyError("arming phrase not given; staying disarmed")
        self.armed = True

    def _do(self, name: str, fn: Optional[Callable] = None, **kw):
        self.calls.add(call=name, **kw)
        if self.dry_run:
            return None
        if not self.armed:
            raise SafetyError("instrument not armed")
        return fn()

    # -- actions ---------------------------------------------------------------
    def mill(self, thickness_nm: float) -> None:
        self.check(thickness_nm=thickness_nm)
        i = 1 if self.axis == "y" else 0
        self.face[i] += self.sign * thickness_nm / 1e3
        cx, cy = self.face
        w, h, d = self.trench_w * 1e-6, thickness_nm * 1e-9, self.mill_d * 1e-6

        def run():
            p = self.m.patterning
            p.set_default_beam_type(self.ION)
            if self.current:
                self.m.beams.ion_beam.beam_current.value = self.current
            p.clear_patterns()
            p.mode = "Serial"
            p.create_cleaning_cross_section(center_x=cx * 1e-6, center_y=cy * 1e-6,
                                            width=w, height=h, depth=d)
            p.run()

        t0 = time.time()
        self._do("patterning.create_cleaning_cross_section+run", run, center_um=(cx, cy),
                 width_m=w, height_m=h, depth_m=d)
        self.clock_s += time.time() - t0
        self.depth_nm += thickness_nm
        self.slices += 1

    def image(self, pixel_nm: float, dwell_us: float) -> np.ndarray:
        self.check(dwell_us=dwell_us)
        w, h = self.resolution
        hfw = pixel_nm * 1e-9 * w

        def run():
            from autoscript_sdb_microscope_client.structures import GrabFrameSettings
            self.m.imaging.set_active_view(self.ELECTRON)
            self.m.imaging.set_active_device(self.ELECTRON)
            self.m.beams.electron_beam.horizontal_field_width.value = hfw
            img = self.m.imaging.grab_frame(GrabFrameSettings(resolution=f"{w}x{h}",
                                                              dwell_time=dwell_us * 1e-6))
            return np.asarray(img.data)

        t0 = time.time()
        out = self._do("imaging.grab_frame", run, hfw_m=hfw, dwell_s=dwell_us * 1e-6,
                       resolution=f"{w}x{h}")
        self.clock_s += time.time() - t0 if not self.dry_run else w * h * dwell_us * 1e-6
        return out if out is not None else np.zeros((h, w), np.uint8)

    def autofocus(self) -> None:
        self._do("auto_functions.run_auto_focus", lambda: self.m.auto_functions.run_auto_focus())

    def auto_contrast(self) -> None:
        self._do("auto_functions.run_auto_cb", lambda: self.m.auto_functions.run_auto_cb())

    def beam_shift(self, dy_nm: float, dx_nm: float) -> None:
        self.shift += np.array([dy_nm, dx_nm]) * 1e-9

        def run():
            from autoscript_sdb_microscope_client.structures import Point
            self.m.beams.electron_beam.beam_shift.value = Point(float(self.shift[1]),
                                                                float(self.shift[0]))

        self._do("beams.electron_beam.beam_shift", run, shift_m=self.shift.tolist())

    def eds(self, points_yx: np.ndarray, pixel_nm: float, dwell_ms: float) -> np.ndarray:
        if self.eds_provider is None:
            raise NotImplementedError("EDS is driven through the EDS vendor API; pass "
                                      "eds_provider=callable(points_yx, pixel_nm, dwell_ms)")
        self.calls.add(call="eds", n_points=len(points_yx), dwell_ms=dwell_ms)
        return self.eds_provider(points_yx, pixel_nm, dwell_ms)


class ShadowAutoScript(Instrument):
    """Drive a digital twin while recording the AutoScript calls a real run
    would issue (dry-run). Use it to review an autonomous script before beam
    time: same agent, same decisions, no hardware."""

    def __init__(self, twin: Instrument, autoscript: Optional[AutoScriptFIBSEM] = None):
        self.twin = twin
        self.rec = autoscript or AutoScriptFIBSEM(dry_run=True, limits=twin.limits)
        self.limits = twin.limits

    clock_s = property(lambda self: self.twin.clock_s)
    depth_nm = property(lambda self: self.twin.depth_nm)
    slices = property(lambda self: self.twin.slices)
    ledger = property(lambda self: getattr(self.twin, "ledger", {}))

    def mill(self, thickness_nm):
        self.rec.mill(thickness_nm)
        self.twin.mill(thickness_nm)

    def image(self, pixel_nm, dwell_us):
        self.rec.image(pixel_nm, dwell_us)
        return self.twin.image(pixel_nm, dwell_us)

    def autofocus(self):
        self.rec.autofocus()
        self.twin.autofocus()

    def auto_contrast(self):
        self.rec.auto_contrast()
        self.twin.auto_contrast()

    def beam_shift(self, dy_nm, dx_nm):
        self.rec.beam_shift(dy_nm, dx_nm)
        self.twin.beam_shift(dy_nm, dx_nm)

    def eds(self, points_yx, pixel_nm, dwell_ms):
        self.rec.calls.add(call="eds (vendor API)", n_points=len(points_yx), dwell_ms=dwell_ms)
        return self.twin.eds(points_yx, pixel_nm, dwell_ms)

    def stop_requested(self):
        return self.twin.stop_requested()
