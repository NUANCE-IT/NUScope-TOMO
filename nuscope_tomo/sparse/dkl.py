"""Deep kernel learning (Wilson et al. 2016; used for autonomous microscopy by
Kalinin, Ziatdinov, Roccapriore et al.): a small CNN maps an image patch to a
low-dimensional embedding, and an exact Gaussian process with an ARD RBF
kernel works in that embedding. Network and kernel are trained jointly by
maximising the GP marginal likelihood, so the model learns *which structural
context predicts the measured property*, and its posterior variance says
where it is still unsure — which is what drives the next measurement.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def extract_patches(img: np.ndarray, centers: np.ndarray, size: int) -> np.ndarray:
    """(N, C, size, size) float32 patches around (y, x) centers; ``img`` is (H, W) or (C, H, W)."""
    img = img[None] if img.ndim == 2 else img
    h = size // 2
    pad = np.pad(img.astype(np.float32), ((0, 0), (h, h), (h, h)), mode="reflect")
    y, x = centers[:, 0] + h, centers[:, 1] + h
    oy, ox = np.meshgrid(np.arange(-h, size - h), np.arange(-h, size - h), indexing="ij")
    return np.stack([c[y[:, None, None] + oy, x[:, None, None] + ox] for c in pad], 1)


class _Encoder(nn.Module):
    def __init__(self, emb: int, cin: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(2),
            nn.Flatten(), nn.Linear(128, emb))

    def forward(self, x):
        return self.net(x)


class DKLRegressor:
    """Multi-output DKL regressor (shared kernel, one GP mean per output)."""

    def __init__(self, emb: int = 6, iters: int = 250, lr: float = 1e-2, seed: int = 0):
        torch.manual_seed(seed)
        self.emb = emb
        self.enc = None  # built on first fit (number of context channels)
        self.log_ls = nn.Parameter(torch.zeros(emb, dtype=torch.float64))
        self.log_os = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_noise = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        self.iters, self.lr = iters, lr

    def _k(self, a, b):
        a, b = a / self.log_ls.exp(), b / self.log_ls.exp()
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        return self.log_os.exp() * torch.exp(-0.5 * d2.clamp(min=0))

    def fit(self, X: np.ndarray, Y: np.ndarray, warm: bool = True):
        if self.enc is None:
            self.enc = _Encoder(self.emb, X.shape[1]).double()
        # per-channel standardisation of the context
        self.xm = X.mean((0, 2, 3), keepdims=True)
        self.xs = X.std((0, 2, 3), keepdims=True) + 1e-6
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-9
        x = torch.from_numpy((X - self.xm) / self.xs).double()
        y = torch.from_numpy((Y - self.ym) / self.ys).double()
        params = list(self.enc.parameters()) + [self.log_ls, self.log_os, self.log_noise]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        n = len(x)
        eye = torch.eye(n, dtype=torch.float64)
        best, best_state = np.inf, None
        iters = self.iters if not getattr(self, "_fitted", False) else max(60, self.iters // 3)
        for _ in range(iters):  # warm start: later rounds continue from the previous fit
            z = self.enc(x)
            K = self._k(z, z) + (self.log_noise.exp().clamp(min=1e-4) + 1e-6) * eye
            try:
                L = torch.linalg.cholesky(K)
            except RuntimeError:  # numerical breakdown: fall back to the best state so far
                break
            alpha = torch.cholesky_solve(y, L)
            nll = (0.5 * (y * alpha).sum() + y.shape[1] * torch.log(torch.diagonal(L)).sum()) \
                / (n * y.shape[1])
            if float(nll) < best:
                best = float(nll)
                best_state = [q.detach().clone() for q in params]
            opt.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        if best_state is not None:  # keep the best marginal likelihood seen
            with torch.no_grad():
                for q, b in zip(params, best_state):
                    q.copy_(b)
        self._fitted = True
        self.nll = best
        with torch.no_grad():
            self.z = self.enc(x)
            K = self._k(self.z, self.z) + (self.log_noise.exp().clamp(min=1e-4) + 1e-6) * eye
            self.L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve(y, self.L)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch: int = 4096):
        mus, vs = [], []
        for i in range(0, len(X), batch):
            x = torch.from_numpy((X[i:i + batch] - self.xm) / self.xs).double()
            zq = self.enc(x)
            Ks = self._k(zq, self.z)
            mu = Ks @ self.alpha
            v = torch.cholesky_solve(Ks.T, self.L)
            var = (self.log_os.exp() - (Ks * v.T).sum(1)).clamp(min=1e-12)
            mus.append(mu.numpy())
            vs.append(var.numpy())
        mu = np.concatenate(mus) * self.ys + self.ym
        var = np.concatenate(vs)  # latent variance in standardised units (shared)
        return mu, var
