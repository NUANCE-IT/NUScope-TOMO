"""U-Net segmentation trained from sparse annotations (scribbles).

Sparse scribbles alone make a U-Net memorise the few labelled pixels (they
sit inside regions, never on boundaries). So by default the scribbles first
train the random forest, whose confident predictions on a set of slices
become dense, probability-weighted pseudo-labels for the U-Net
(distillation); the network then adds the spatial context the per-pixel
forest lacks. With ``pseudo_labels=False`` the loss uses scribbled pixels only. Training patches are
centred on annotated pixels; flips/rotations and brightness jitter augment
them. Inference runs slice by slice over the whole stack.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..core.registry import op
from ..core.volume import Volume
from ..segment.classifier import auto_annotations, scribbles_from_json
from .unet import UNet2D, device, predict_slices


def pseudo_label(image: Volume, annotations: Volume, classes: int, n_slices: int = 12,
                 confidence: float = 0.6) -> tuple[Volume, np.ndarray]:
    """Dense RF pseudo-labels (0 where unsure) and per-pixel weights on n slices."""
    from ..segment.classifier import PixelClassifier
    clf = PixelClassifier(n_estimators=60).fit(image, annotations)
    zs = np.unique(np.linspace(0, image.shape[0] - 1, n_slices).astype(int))
    lab = np.zeros(image.shape, np.uint8)
    wts = np.zeros(image.shape, np.float32)
    for z in zs:
        pr = clf.predict_slice(image.get_slice(0, int(z)), proba=True)
        c = clf.rf.classes_[np.argmax(pr, -1)]
        pm = pr.max(-1)
        keep = pm >= confidence
        lab[z][keep] = c[keep]
        wts[z] = np.where(keep, pm, 0)
    ann = np.asarray(annotations.data)
    has = ann > 0  # the user's own strokes always win, at full weight
    lab[has], wts[has] = ann[has], 1.0
    return image.derive(lab, kind="labels"), wts


def _stack(image: Volume, z: int, context: int) -> np.ndarray:
    """(2*context+1, H, W) slices centred on z (edge-clamped)."""
    n = image.shape[0]
    return np.stack([image.get_slice(0, int(np.clip(z + d, 0, n - 1)))
                     for d in range(-context, context + 1)]).astype(np.float32)


def train_unet(image: Volume, annotations: Volume, classes: int, steps: int = 2000,
               patch: int = 128, batch: int = 16, lr: float = 1e-3, seed: int = 0,
               weights: Optional[np.ndarray] = None, context: int = 1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = device()
    ann = annotations.data
    zs = [int(z) for z in np.flatnonzero(np.asarray((ann > 0).any(axis=(1, 2))))]
    imgs = {z: _stack(image, z, context) for z in zs}
    labs = {z: np.asarray(ann[z]).astype(np.int64) for z in zs}
    pw = {z: (weights[z] if weights is not None else (labs[z] > 0).astype(np.float32))
          for z in zs}
    allpix = np.concatenate([image.get_slice(0, z).ravel()[::17] for z in zs]).astype(np.float32)
    mu, sd = float(allpix.mean()), float(allpix.std() + 1e-6)
    coords = [(z, y, x) for z in zs for y, x in np.argwhere(labs[z] > 0)[::97]]
    h, w = image.shape[1:]
    half = patch // 2

    def get_batch():
        C = 2 * context + 1
        X = np.empty((batch, C, patch, patch), np.float32)
        Y = np.zeros((batch, patch, patch), np.int64)  # 0 = unlabelled (ignored)
        Wt = np.zeros((batch, patch, patch), np.float32)
        for i in range(batch):
            z, y, x = coords[rng.integers(len(coords))]
            y0 = int(np.clip(y - half + rng.integers(-half // 2, half // 2 + 1), 0, h - patch))
            x0 = int(np.clip(x - half + rng.integers(-half // 2, half // 2 + 1), 0, w - patch))
            im = (imgs[z][:, y0:y0 + patch, x0:x0 + patch] - mu) / sd
            lb = labs[z][y0:y0 + patch, x0:x0 + patch]
            wt = pw[z][y0:y0 + patch, x0:x0 + patch]
            k = rng.integers(4)
            im, lb, wt = np.rot90(im, k, (1, 2)), np.rot90(lb, k), np.rot90(wt, k)
            if rng.random() < 0.5:
                im, lb, wt = im[:, :, ::-1], lb[:, ::-1], wt[:, ::-1]
            if rng.random() < 0.5:  # milling direction is symmetric for context
                im = im[::-1]
            X[i] = im * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05)
            Y[i], Wt[i] = lb, wt
        return (torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev),
                torch.from_numpy(Wt).to(dev))

    model = UNet2D(2 * context + 1, classes, base=32, depth=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    counts = np.bincount(np.concatenate([labs[z].ravel() for z in zs]), minlength=classes + 1)[1:]
    wts = torch.tensor(counts.sum() / np.maximum(counts, 1) / classes, dtype=torch.float32,
                       device=dev)
    t0, hist = time.time(), []
    model.train()
    for it in range(steps):
        X, Y, Wt = get_batch()
        logits = model(X)
        per_px = F.cross_entropy(logits, (Y - 1).clamp(min=0), weight=wts, reduction="none")
        loss = (per_px * Wt).sum() / Wt.sum().clamp(min=1e-6)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
    info = {"steps": steps, "seconds": round(time.time() - t0, 1), "device": str(dev),
            "context_slices": context,
            "final_loss": float(np.mean(hist[-100:])), "train_slices": zs,
            "annotated_pixels": int(counts.sum()), "mean": mu, "std": sd}
    return model, info, hist


@op("segment.unet", inputs=("volume", "annotations"))
def unet_segment(volume: Volume, annotations: Optional[Volume] = None,
                 scribbles: Optional[str] = None, classes: int = 3,
                 class_names: Optional[dict] = None,
                 bands: Optional[Sequence[Sequence[float]]] = None, n_slices: int = 10,
                 steps: int = 2000, pseudo_labels: bool = True, context: int = 1,
                 model_path: Optional[str] = None) -> dict:
    """Deep-learning (U-Net) segmentation trained from sparse scribbles."""
    if annotations is None:
        annotations = (volume.derive(scribbles_from_json(scribbles, volume.shape), kind="labels")
                       if scribbles else auto_annotations(volume, classes=classes, bands=bands,
                                                          n_slices=n_slices))
    weights = None
    if pseudo_labels:
        annotations, weights = pseudo_label(volume, annotations, classes)
    model, info, hist = train_unet(volume, annotations, classes, steps=steps, weights=weights,
                                   context=context)
    info["pseudo_labels"] = pseudo_labels
    if model_path:
        torch.save({"state": model.state_dict(), "info": info, "classes": class_names},
                   model_path)
    mu, sd = info["mean"], info["std"]

    def block(b, block_info=None):
        z0, z1 = block_info[0]["array-location"][0]
        stacks = np.stack([_stack(volume, z, context) for z in range(z0, z1)])
        logits = predict_slices(model, (stacks - mu) / sd)
        return (np.argmax(logits, 1) + 1).astype(np.uint8)

    lab = volume.data.rechunk((16, -1, -1)).map_blocks(block, dtype=np.uint8)
    names = class_names or {i: f"class_{i}" for i in range(1, classes + 1)}
    out = volume.derive(lab, name=f"{volume.name}_unet", kind="labels",
                        classes={str(k): v for k, v in names.items()},
                        classifier={"type": "unet", **{k: v for k, v in info.items()
                                                       if k not in ("mean", "std")}})
    return {"volume": out,
            "training": pd.DataFrame({"step": np.arange(len(hist)), "loss": hist})}


__all__ = ["train_unet", "unet_segment"]
