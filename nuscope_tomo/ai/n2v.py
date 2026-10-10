"""Noise2Void (Krull et al., CVPR 2019): self-supervised denoising.

Pixel noise is assumed independent between pixels while the signal is not,
so a network that predicts a pixel from its *neighbours only* (the pixel is
masked out) learns the signal. No clean target images are needed: the model
is trained on the stack being denoised.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from ..core.registry import op
from ..core.volume import Volume
from ..preprocess._util import to_dtype
from .unet import UNet2D, device, predict_slices


def _sample_patches(vol: Volume, n: int, size: int, rng) -> np.ndarray:
    nz, ny, nx = vol.shape
    zs = rng.integers(0, nz, n)
    out = np.empty((n, size, size), np.float32)
    cache = {}
    for i, z in enumerate(zs):
        if z not in cache:
            cache[z] = vol.get_slice(0, int(z)).astype(np.float32)
        s = cache[z]
        y, x = rng.integers(0, ny - size + 1), rng.integers(0, nx - size + 1)
        p = s[y:y + size, x:x + size]
        k = rng.integers(0, 4)
        p = np.rot90(p, k)
        if rng.random() < 0.5:
            p = p[:, ::-1]
        out[i] = p
    return out


def _mask_blind_spots(batch: torch.Tensor, frac: float, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of pixels by a random neighbour (blind spots)."""
    b, _, h, w = batch.shape
    m = torch.from_numpy(rng.random((b, 1, h, w)) < frac).to(batch.device)
    dy = torch.from_numpy(rng.integers(-2, 3, (b, 1, h, w))).to(batch.device)
    dx = torch.from_numpy(rng.integers(-2, 3, (b, 1, h, w))).to(batch.device)
    zero = (dy == 0) & (dx == 0)
    dx = torch.where(zero, torch.ones_like(dx), dx)
    yy = torch.arange(h, device=batch.device).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.arange(w, device=batch.device).view(1, 1, 1, w).expand(b, 1, h, w)
    ys = (yy + dy).clamp(0, h - 1)
    xs = (xx + dx).clamp(0, w - 1)
    flat = batch.view(b, -1)
    idx = (ys * w + xs).view(b, -1)
    neigh = torch.gather(flat, 1, idx).view_as(batch)
    return torch.where(m, neigh, batch), m


def train_n2v(vol: Volume, steps: int = 1500, patch: int = 64, batch: int = 64,
              frac: float = 0.02, lr: float = 4e-4, seed: int = 0,
              log: Optional[Callable[[str], None]] = None) -> tuple[UNet2D, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = device()
    sample = np.asarray(vol.data[:: max(1, vol.shape[0] // 32)]).astype(np.float32)
    mu, sd = float(sample.mean()), float(sample.std() + 1e-6)
    model = UNet2D(1, 1, base=32, depth=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pool = (_sample_patches(vol, 4096, patch, rng) - mu) / sd
    t0, hist = time.time(), []
    model.train()
    for it in range(steps):
        if it and it % 500 == 0:  # refresh the patch pool
            pool = (_sample_patches(vol, 4096, patch, rng) - mu) / sd
        x = torch.from_numpy(pool[rng.integers(0, len(pool), batch)][:, None]).to(dev)
        xin, m = _mask_blind_spots(x, frac, rng)
        loss = ((model(xin) - x) ** 2)[m].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
        if log and it % 250 == 0:
            log(f"n2v step {it}/{steps} loss {np.mean(hist[-50:]):.4f}")
    info = {"steps": steps, "seconds": round(time.time() - t0, 1), "device": str(dev),
            "final_loss": float(np.mean(hist[-100:])), "mean": mu, "std": sd}
    return model, info


@op("preprocess.denoise_n2v")
def denoise_n2v(volume: Volume, steps: int = 1500, patch: int = 64, batch: int = 64,
                model_path: Optional[str] = None) -> Volume:
    """Noise2Void self-supervised deep-learning denoising (trained on this stack)."""
    model, info = train_n2v(volume, steps, patch, batch)
    if model_path:
        torch.save({"state": model.state_dict(), "info": info}, model_path)
    mu, sd = info["mean"], info["std"]
    arr = volume.data.rechunk((16, -1, -1))

    def block(b):
        y = predict_slices(model, (b.astype(np.float32) - mu) / sd)[:, 0] * sd + mu
        return to_dtype(y, b.dtype)

    out = arr.map_blocks(block, dtype=volume.dtype)
    return volume.derive(out, denoise={"method": "noise2void", **info})


__all__ = ["train_n2v", "denoise_n2v"]
