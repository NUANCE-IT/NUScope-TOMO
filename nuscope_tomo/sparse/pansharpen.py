"""Coupled-NMF pan-sharpening for correlative chemical imaging, after
Borodinov, Kalinin, Ovchinnikova et al., npj Comput. Mater. 5, 49 (2019)
(AFM-IR), with the hyperspectral pan-sharpening formulation of Yokoya et al.
(IEEE TGRS 2012) and Loncan et al. (IEEE GRSM 2015).

Two partial views of the same unknown high-resolution chemical cube
X (n pixels × E channels):

* ``Y_H`` — a *low-spatial-resolution* map with the full chemical vector
  (coarse EDS hypermap: every element, few pixels);
* ``Y_M`` — *high-resolution* channels that are known linear functions of
  the chemistry, Y_M ≈ X Rᵀ. In FIB-SEM: the BSE image (η = Σ cᵢ ηᵢ,
  Castaing's rule) and optionally a few fast, short-dwell element maps.

With X = U H (U ≥ 0 abundances, H ≥ 0 endmembers): NMF of Y_H gives H;
the abundances are initialised by upsampling the low-resolution ones and then
refined by multiplicative updates on

    ‖Y_M − U H Rᵀ‖² + μ ‖Y_H − D(U) H‖²,

where D is the spatial block-averaging of the coarse acquisition. Because
R encodes physics (not a fitted correlation), the fusion does not invent
chemistry from contrast that has no chemical origin — the point Borodinov
et al. make about SE/optical fusion.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import NMF

# Backscatter coefficients (Reuter 1972 fit; 20 kV-ish), by element symbol
_Z = {"C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
      "S": 16, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
      "Zn": 30, "Y": 39, "Zr": 40, "La": 57, "W": 74, "Pt": 78, "Au": 79}


def bse_coefficient(element: str) -> float:
    Z = _Z[element]
    return -0.0254 + 0.016 * Z - 1.86e-4 * Z ** 2 + 8.3e-7 * Z ** 3


def bse_response(elements: list[str]) -> np.ndarray:
    """Row of R mapping a composition vector to BSE yield (Castaing's rule)."""
    return np.array([[bse_coefficient(e) for e in elements]])


def _down(U: np.ndarray, shape_hi, B: int) -> np.ndarray:
    H, W = shape_hi
    k = U.shape[1]
    return U.reshape(H // B, B, W // B, B, k).mean((1, 3)).reshape(-1, k)


def _up_adj(X: np.ndarray, shape_hi, B: int) -> np.ndarray:
    """Adjoint of block averaging: replicate and divide by B²."""
    H, W = shape_hi
    k = X.shape[1]
    Xc = X.reshape(H // B, W // B, k) / (B * B)
    return np.repeat(np.repeat(Xc, B, 0), B, 1).reshape(-1, k)


def cnmf_pansharpen(Y_H: np.ndarray, Y_M: np.ndarray, R: np.ndarray, B: int,
                    n_components: int = 4, mu: float = 1.0, iters: int = 400,
                    seed: int = 0, channel_weights: Optional[np.ndarray] = None) -> dict:
    """Fuse a coarse full-chemistry map with high-resolution linked channels.

    Y_H : (h, w, E) coarse map;  Y_M : (h*B, w*B, M) high-res channels;
    R   : (M, E) known linear response of each high-res channel to chemistry.
    Returns the high-resolution cube (H, W, E), endmembers (k, E), abundances.
    """
    h, w, E = Y_H.shape
    Hh, Wh, M = Y_M.shape
    assert (Hh, Wh) == (h * B, w * B), "Y_M must be exactly B× the Y_H grid"
    YH = np.clip(Y_H.reshape(-1, E), 0, None)
    YM = Y_M.reshape(-1, M).astype(np.float64)
    # 1. endmembers from the coarse full-chemistry map
    nmf = NMF(n_components, init="nndsvda", max_iter=2000, random_state=seed)
    UH = nmf.fit_transform(YH)
    Hc = nmf.components_                        # (k, E)
    # 2. initial high-res abundances: bilinear upsampling of the coarse ones
    from scipy.ndimage import zoom
    U = np.stack([zoom(UH[:, j].reshape(h, w), B, order=1) for j in range(n_components)], -1)
    U = np.clip(U.reshape(-1, n_components), 1e-9, None)
    # 3. channel scaling so each high-res channel carries comparable weight
    HM = Hc @ R.T                                # (k, M) endmember response in Y_M
    s = YM.std(0) + 1e-12
    wts = channel_weights if channel_weights is not None else np.ones(M)
    YMs, HMs = YM / s * wts, HM / s * wts
    sh = YH.std() + 1e-12
    YHs, Hcs = YH / sh, Hc / sh
    # 4. multiplicative updates on the coupled objective (abundances only)
    A = HMs @ HMs.T
    C = Hcs @ Hcs.T
    num_fixed = YMs @ HMs.T + mu * _up_adj(YHs @ Hcs.T, (Hh, Wh), B)
    for _ in range(iters):
        den = U @ A + mu * _up_adj(_down(U, (Hh, Wh), B) @ C, (Hh, Wh), B) + 1e-12
        U *= num_fixed / den
    X = (U @ Hc).reshape(Hh, Wh, E)
    return {"cube": X, "endmembers": Hc, "abundances": U.reshape(Hh, Wh, n_components),
            "coarse_abundances": UH.reshape(h, w, n_components)}


def fusion_metrics(est: np.ndarray, truth: np.ndarray) -> dict:
    """Pan-sharpening quality metrics used by Borodinov et al.: cross-correlation
    (geometry, 1 = perfect), spectral angle mapper (radians, 0 = perfect), RMSE."""
    cc = np.mean([np.corrcoef(est[..., e].ravel(), truth[..., e].ravel())[0, 1]
                  for e in range(truth.shape[-1]) if truth[..., e].std() > 0])
    num = (est * truth).sum(-1)
    den = np.linalg.norm(est, axis=-1) * np.linalg.norm(truth, axis=-1) + 1e-12
    sam = float(np.mean(np.arccos(np.clip(num / den, -1, 1))))
    return {"cc": float(cc), "sam": sam, "rmse": float(np.sqrt(np.mean((est - truth) ** 2)))}


__all__ = ["cnmf_pansharpen", "bse_response", "bse_coefficient", "fusion_metrics"]
