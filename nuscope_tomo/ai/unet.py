"""A compact 2D U-Net and device helpers."""
from __future__ import annotations

import threading

import numpy as np
import torch
from torch import nn

_GPU_LOCK = threading.Lock()  # Dask runs blocks in threads; MPS/CUDA calls are serialised


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet2D(nn.Module):
    """U-Net with ``depth`` down-sampling levels; input size must be divisible by 2**depth."""

    def __init__(self, cin: int = 1, cout: int = 1, base: int = 32, depth: int = 3,
                 residual: bool = False):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(cin, ch[0])] +
                                  [_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], cout, 1)
        self.pool = nn.MaxPool2d(2)
        self.residual = residual and cin == cout
        self.depth = depth

    def forward(self, x):
        skips = []
        h = x
        for i, d in enumerate(self.down):
            h = d(h if i == 0 else self.pool(h))
            skips.append(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            h = dec(torch.cat([up(h), skip], 1))
        out = self.head(h)
        return out + x if self.residual else out


def pad_to(img: np.ndarray, mult: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    pad = [(0, 0)] * (img.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(img, pad, mode="reflect"), (h, w)


@torch.no_grad()
def predict_slices(model: nn.Module, slices: np.ndarray, batch: int = 8) -> np.ndarray:
    """Run a 2D model over (N, H, W) or (N, Cin, H, W) float32 slices → (N, C, H, W)."""
    with _GPU_LOCK:
        return _predict(model, slices, batch)


def _predict(model, slices, batch):
    dev = next(model.parameters()).device
    model.eval()
    mult = 2 ** model.depth
    out = []
    for i in range(0, len(slices), batch):
        x, (h, w) = pad_to(slices[i:i + batch], mult)
        x = x[:, None] if x.ndim == 3 else x
        t = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        out.append(model(t)[..., :h, :w].float().cpu().numpy())
    return np.concatenate(out)
