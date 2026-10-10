"""Machine-learning segmentation from sparse annotations ("scribbles").

The Amira/ilastik-style workflow: the user paints a few strokes per class on
a handful of slices, a random forest learns from a bank of multi-scale image
features, then every voxel of the stack is classified out-of-core, slice by
slice. Feature computation is 2D per FIB slice (the XY resolution is usually
much finer than the slice thickness) with optional neighbouring-slice context.

Annotations are an integer volume of the same shape (0 = unlabelled,
1..K = class), e.g. painted in NuScope Studio and saved with the project.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, structure_tensor, \
    structure_tensor_eigenvalues
from sklearn.ensemble import RandomForestClassifier

from ..core.registry import op
from ..core.volume import Volume
from ..preprocess._util import slicewise

DEFAULT_SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def feature_names(sigmas: Sequence[float] = DEFAULT_SIGMAS, context: bool = False) -> list[str]:
    names = ["intensity"]
    for s in sigmas:
        names += [f"gauss_{s}", f"log_{s}", f"gradmag_{s}", f"hess1_{s}", f"hess2_{s}",
                  f"st1_{s}", f"st2_{s}", f"dog_{s}"]
    if context:
        names += ["z_prev_gauss2", "z_next_gauss2"]
    return names


def slice_features(img: np.ndarray, sigmas: Sequence[float] = DEFAULT_SIGMAS) -> np.ndarray:
    """(H, W, F) float32 feature stack for one slice."""
    a = img.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    feats = [a]
    prev = a
    for s in sigmas:
        g = ndi.gaussian_filter(a, s)
        feats.append(g)
        feats.append(ndi.gaussian_laplace(a, s) * s * s)
        feats.append(ndi.gaussian_gradient_magnitude(a, s) * s)
        H = hessian_matrix(a, sigma=s, order="rc", use_gaussian_derivatives=False)
        e = hessian_matrix_eigvals(H)
        feats += [e[0] * s * s, e[1] * s * s]
        S = structure_tensor(a, sigma=s, order="rc")
        t = structure_tensor_eigenvalues(S)
        feats += [t[0], t[1]]
        feats.append(g - prev)
        prev = g
    return np.stack(feats, -1).astype(np.float32)


class PixelClassifier:
    """Random-forest voxel classifier on multi-scale features."""

    def __init__(self, sigmas: Sequence[float] = DEFAULT_SIGMAS, n_estimators: int = 100,
                 max_depth: Optional[int] = 16, class_names: Optional[dict] = None):
        self.sigmas = tuple(sigmas)
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         n_jobs=-1, class_weight="balanced_subsample",
                                         random_state=0)
        self.class_names = class_names or {}

    def fit(self, image: Volume, annotations: Volume, max_samples: int = 200_000):
        ann = annotations.data
        zs = [int(z) for z in np.flatnonzero(np.asarray((ann > 0).any(axis=(1, 2))))]
        if not zs:
            raise ValueError("No annotated voxels")
        X, y = [], []
        rng = np.random.default_rng(0)
        for z in zs:
            lab = np.asarray(ann[z])
            m = lab > 0
            F = slice_features(image.get_slice(0, z), self.sigmas)
            X.append(F[m])
            y.append(lab[m])
        X, y = np.concatenate(X), np.concatenate(y)
        if len(y) > max_samples:  # stratified subsample
            keep = []
            for c in np.unique(y):
                idx = np.flatnonzero(y == c)
                keep.append(rng.choice(idx, min(len(idx), max_samples // len(np.unique(y))),
                                       replace=False))
            keep = np.concatenate(keep)
            X, y = X[keep], y[keep]
        self.rf.fit(X, y)
        self.n_train = int(len(y))
        self.train_slices = zs
        return self

    def importances(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": feature_names(self.sigmas),
                              "importance": self.rf.feature_importances_})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    def predict_slice(self, img: np.ndarray, proba: bool = False) -> np.ndarray:
        F = slice_features(img, self.sigmas)
        h, w, f = F.shape
        flat = F.reshape(-1, f)
        if proba:
            return self.rf.predict_proba(flat).reshape(h, w, -1)
        return self.rf.predict(flat).reshape(h, w).astype(np.uint8)

    def predict(self, image: Volume) -> Volume:
        lab = slicewise(image, self.predict_slice, dtype=np.uint8, z_chunk=4)
        return image.derive(lab, name=f"{image.name}_classes", kind="labels",
                            classes={str(k): v for k, v in self.class_names.items()},
                            classifier={"type": "random_forest", "sigmas": list(self.sigmas),
                                        "n_train": self.n_train,
                                        "train_slices": self.train_slices})

    def save(self, path: str | Path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "PixelClassifier":
        return pickle.loads(Path(path).read_bytes())


def scribbles_from_json(spec: str | Path | list, shape: Sequence[int]) -> np.ndarray:
    """Rasterise brush strokes ``[{"class": k, "z": z, "points": [[y, x], ...],
    "radius": r}, ...]`` into a sparse annotation array."""
    from skimage.draw import disk, line

    strokes = json.loads(Path(spec).read_text()) if isinstance(spec, (str, Path)) else spec
    ann = np.zeros(shape, np.uint8)
    for s in strokes:
        z, r, c = int(s["z"]), float(s.get("radius", 3)), int(s["class"])
        pts = np.asarray(s["points"], float)
        mask = np.zeros(shape[1:], bool)
        for p0, p1 in zip(pts[:-1], pts[1:]) if len(pts) > 1 else [(pts[0], pts[0])]:
            rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
            mask[np.clip(rr, 0, shape[1] - 1), np.clip(cc, 0, shape[2] - 1)] = True
        if r > 1:
            mask = ndi.binary_dilation(mask, iterations=int(r))
        ann[z][mask] = c
    return ann


def auto_annotations(image: Volume, classes: int = 3, n_slices: int = 6,
                     erode_px: int = 6, fraction: float = 0.02,
                     bands: Optional[Sequence[Sequence[float]]] = None,
                     smooth_px: float = 1.5) -> Volume:
    """Bootstrap sparse annotations from confident intensity cores.

    Gives the forest a starting point without manual painting: on a few
    slices, voxels deep inside each class (after erosion) are sampled.
    Classes come from multi-Otsu, or from explicit ``bands`` =
    [[lo, hi], ...] per class (earlier bands win where they overlap) — the
    forest then learns the texture/context that intensity alone cannot.
    """
    from skimage.filters import threshold_multiotsu

    rng = np.random.default_rng(1)
    if bands is None:
        s = np.asarray(image.data[::4, ::4, ::4]).ravel()
        t = threshold_multiotsu(s[s > 0], classes=classes)
        edges = [-np.inf, *t, np.inf]
        bands = [(edges[i], edges[i + 1]) for i in range(classes)]
    ann = np.zeros(image.shape, np.uint8)
    for z in np.linspace(0, image.shape[0] - 1, n_slices + 2)[1:-1].astype(int):
        sl = ndi.gaussian_filter(image.get_slice(0, z).astype(np.float32), smooth_px)
        taken = np.zeros(sl.shape, bool)
        for c, (lo, hi) in enumerate(bands, 1):
            m = (sl >= lo) & (sl <= hi) & ~taken
            taken |= m
            core = ndi.binary_erosion(m, iterations=erode_px) if erode_px else m
            frac = fraction if core.sum() > 2000 else 1.0  # keep all of rare classes
            ann[z][core & (rng.random(core.shape) < frac)] = c
    return image.derive(ann, kind="labels", name="auto_annotations")


@op("segment.random_forest", inputs=("volume", "annotations"))
def random_forest(volume: Volume, annotations: Optional[Volume] = None,
                  scribbles: Optional[str] = None, classes: int = 3,
                  class_names: Optional[dict] = None,
                  bands: Optional[Sequence[Sequence[float]]] = None, n_slices: int = 6,
                  sigmas: Sequence[float] = DEFAULT_SIGMAS, n_estimators: int = 100,
                  model_path: Optional[str] = None) -> dict:
    """Train a random-forest classifier on sparse annotations and segment all slices."""
    if annotations is None:
        annotations = (volume.derive(scribbles_from_json(scribbles, volume.shape), kind="labels")
                       if scribbles else auto_annotations(volume, classes=classes, bands=bands,
                                         n_slices=n_slices))
    clf = PixelClassifier(sigmas, n_estimators,
                          class_names=class_names or {i: f"class_{i}"
                                                      for i in range(1, classes + 1)})
    clf.fit(volume, annotations)
    if model_path:
        clf.save(model_path)
    return {"volume": clf.predict(volume), "feature_importance": clf.importances(),
            "model": clf}


__all__ = ["PixelClassifier", "slice_features", "feature_names", "scribbles_from_json",
           "auto_annotations", "random_forest"]
