"""Live acquisition mode (not available in Avizo): analyse slices while the
FIB is still milling.

``nuscope-tomo watch <folder>`` follows the folder Auto Slice & View writes to.
Each complete new image is cropped, destriped, shading-corrected,
brightness-matched, binned, denoised and (optionally) classified with a
trained model, then appended to growing Zarr stores in ``<project>/live``.
Running phase fractions with 95 % confidence intervals (integral range) are
written to ``live.json``; when every phase reaches the requested precision
the session reports that milling could stop. Studio shows the growing
volume live.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import zarr
from scipy import ndimage as ndi

from .core.volume import META_FILE, Volume, VoxelSize, _jsonable
from .io.stack import list_slices, read_image
from .preprocess.correct import destripe_slice
from .preprocess.geometry import detect_sample_roi


class LiveSession:
    def __init__(self, folder: str | Path, project: str | Path, voxel_size: VoxelSize,
                 model_path: Optional[str | Path] = None, warmup: int = 8, bin_xy: int = 2,
                 denoise_h: float = 0.8, target_rel_error: float = 0.05,
                 stats_every: int = 25, log: Callable[[str], None] = print):
        self.folder, self.root = Path(folder), Path(project) / "live"
        self.root.mkdir(parents=True, exist_ok=True)
        self.vs, self.warmup, self.bin = voxel_size, warmup, bin_xy
        self.h, self.target, self.every, self.log = denoise_h, target_rel_error, stats_every, log
        self.model = pickle.loads(Path(model_path).read_bytes()) if model_path else None
        self.seen: dict[str, int] = {}
        self.done: list[str] = []
        self.pending: list[np.ndarray] = []
        self.pending_all: Optional[list] = None
        self.roi = self.shading = None
        self.window: list[np.ndarray] = []  # recent cropped raw slices (¼ resolution)
        self.ref_pct = None
        self.pct_hist: list[np.ndarray] = []
        self.counts = None
        self.history: list[dict] = []
        self.t0 = time.time()
        self.stores: dict[str, zarr.Array] = {}

    # ---------------------------------------------------------------- stores
    def _store(self, name: str, shape_yx, dtype, vs: VoxelSize, kind: str, meta=None):
        if name not in self.stores:
            path = self.root / f"{name}.zarr"
            arr = zarr.create_array(str(path), shape=(0, *shape_yx), chunks=(8, *shape_yx),
                                    dtype=dtype, overwrite=True)
            (path / META_FILE).write_text(json.dumps(_jsonable(
                {"name": f"live_{name}", "kind": kind, "voxel_size": vs.__dict__,
                 "meta": meta or {}}), indent=2))
            self.stores[name] = arr
        return self.stores[name]

    def _append(self, name: str, img: np.ndarray, **kw):
        arr = self._store(name, img.shape, img.dtype, **kw)
        n = arr.shape[0]
        arr.resize((n + 1, *img.shape))
        arr[n] = img

    # ------------------------------------------------------------ discovery
    def new_files(self) -> list[Path]:
        try:
            files = list_slices(self.folder)
        except FileNotFoundError:
            return []
        ready = []
        for f in files:
            if f.name in self.done:
                continue
            size = f.stat().st_size
            if self.seen.get(f.name) == size and size > 0:  # unchanged since last poll
                ready.append(f)
            self.seen[f.name] = size
        return ready

    # ----------------------------------------------------------- processing
    def _start(self):
        """Fix the block-face ROI from the first slices."""
        stack = np.stack(self.pending)
        self.roi = detect_sample_roi(Volume(stack, self.vs))
        (y0, y1), (x0, x1) = self.roi["y"], self.roi["x"]
        self.log(f"live: block face y {y0}-{y1}, x {x0}-{x1}")

    def _update_shading(self, crop: np.ndarray):
        """Illumination field from a rolling window of the last 32 slices.

        Shading changes as the trench deepens, so a field fixed at the start
        goes stale; the median over many slices also averages the
        microstructure out of the estimate.
        """
        self.window.append(crop[::4, ::4].astype(np.float32))
        self.window = self.window[-32:]
        n = len(self.pct_hist)
        if self.shading is None or n % 8 == 0:
            med = np.median(np.stack(self.window), 0)
            f = ndi.gaussian_filter(med, max(med.shape) / 8)
            f = ndi.zoom(f / f.mean(), (crop.shape[0] / f.shape[0], crop.shape[1] / f.shape[1]),
                         order=1)
            self.shading = f[: crop.shape[0], : crop.shape[1]]

    def process(self, raw: np.ndarray):
        self._append("raw", raw, vs=self.vs, kind="intensity")
        (y0, y1), (x0, x1) = self.roi["y"], self.roi["x"]
        crop = raw[y0:y1, x0:x1]
        if len(self.window) < 8 and self.pending_all is not None:
            for p in self.pending_all:  # seed the window with the warm-up slices
                self.window.append(p[y0:y1, x0:x1][::4, ::4].astype(np.float32))
            self.pending_all = None
        self._update_shading(crop)
        img = destripe_slice(crop).astype(np.float32) / self.shading
        p = np.percentile(img[::4, ::4], (2, 98))
        self.pct_hist.append(p)
        ref = np.median(np.array(self.pct_hist[-15:]), 0)
        img = (img - p[0]) * (ref[1] - ref[0]) / max(p[1] - p[0], 1e-3) + ref[0]
        b = self.bin
        h, w = img.shape[0] // b * b, img.shape[1] // b * b
        img = img[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))
        from skimage.restoration import denoise_nl_means, estimate_sigma
        a = np.clip(img, 0, 255) / 255.0
        sig = float(estimate_sigma(a))
        img = denoise_nl_means(a, h=self.h * sig, sigma=sig, patch_size=5, patch_distance=6,
                               fast_mode=True) * 255
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        vs_b = self.vs.scaled(1, b, b)
        self._append("processed", img, vs=vs_b, kind="intensity")
        if self.model is not None:
            lab = self.model.predict_slice(img)
            names = {str(k): v for k, v in self.model.class_names.items()}
            self._append("labels", lab, vs=vs_b, kind="labels", meta={"classes": names})
            k = len(self.model.class_names) + 1
            c = np.bincount(lab.ravel(), minlength=k)
            self.counts = c if self.counts is None else self.counts + c

    def statistics(self) -> dict:
        n = self.stores["processed"].shape[0] if "processed" in self.stores else 0
        out = {"slices": n, "depth_um": n * self.vs.z / 1e3, "elapsed_s": time.time() - self.t0,
               "rate_slices_per_min": 60 * n / max(time.time() - self.t0, 1e-6),
               "phases": [], "can_stop": False}
        if self.model is None or "labels" not in self.stores or n < 16:
            return out
        from .analyze.confidence import integral_range_cov
        lab = Volume(self.stores["labels"][:], self.vs.scaled(1, self.bin, self.bin),
                     kind="labels").preview(4_000_000, isotropic=True)
        arr = lab.numpy()
        V = n * self.vs.z * np.prod(np.array(self.stores["labels"].shape[1:]) *
                                     self.vs.y * self.bin) / 1e9
        ok = True
        for cls, name in self.model.class_names.items():
            m = arr == int(cls)
            p = float(self.counts[int(cls)] / self.counts[1:].sum())
            A3 = integral_range_cov(m, lab.voxel_size.as_tuple()) if 0 < m.mean() < 1 else np.nan
            rel = 2 * np.sqrt(p * (1 - p) * A3 / V) / p if p > 0 and np.isfinite(A3) else np.inf
            reached = rel <= self.target
            if p > 0.01:  # trace phases don't hold up the decision
                ok &= reached
            out["phases"].append({"class": int(cls), "name": name, "fraction": p,
                                  "rel_error_95": rel, "target_reached": bool(reached)})
        out["can_stop"] = bool(ok)
        return out

    def _publish(self, stats: dict):
        self.history.append({k: v for k, v in stats.items() if k != "phases"} |
                            {f"f_{p['name']}": p["fraction"] for p in stats["phases"]} |
                            {f"err_{p['name']}": p["rel_error_95"] for p in stats["phases"]})
        doc = {"status": "running", "updated": time.strftime("%H:%M:%S"),
               "target_rel_error": self.target, **stats}
        (self.root / "live.json").write_text(json.dumps(_jsonable(doc), indent=2, default=str))
        import pandas as pd
        pd.DataFrame(self.history).to_csv(self.root / "live_history.csv", index=False)

    # ----------------------------------------------------------------- loop
    def step(self) -> int:
        files = self.new_files()
        for f in files:
            img = read_image(f)
            self.done.append(f.name)
            if self.roi is None:
                self.pending.append(img)
                if len(self.pending) >= self.warmup:
                    self._start()
                    self.pending_all = list(self.pending)
                    for p in self.pending:
                        self.process(p)
                    self.pending = []
                continue
            self.process(img)
            n = self.stores["processed"].shape[0]
            if n % self.every == 0:
                st = self.statistics()
                self._publish(st)
                fr = ", ".join(f"{p['name']} {100 * p['fraction']:.1f}±{100 * p['rel_error_95'] * p['fraction']:.1f}%"
                               for p in st["phases"])
                self.log(f"live: {n} slices, {st['depth_um']:.2f} µm  {fr}"
                         + ("  → precision target reached, milling can stop" if st["can_stop"] else ""))
        return len(files)

    def run(self, poll: float = 1.0, idle_timeout: float = 60.0):
        idle = 0.0
        while idle < idle_timeout:
            got = self.step()
            idle = 0.0 if got else idle + poll
            time.sleep(poll)
        st = self.statistics()
        self._publish(st)
        doc = json.loads((self.root / "live.json").read_text())
        doc["status"] = "finished"
        (self.root / "live.json").write_text(json.dumps(doc, indent=2))
        self.log(f"live: finished, {st['slices']} slices")
        return st
