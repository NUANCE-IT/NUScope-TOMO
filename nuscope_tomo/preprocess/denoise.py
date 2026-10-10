"""Noise reduction. 2D filters run per FIB slice (out-of-core); 3D filters
run on overlapping z-chunks with physically scaled (anisotropic) kernels."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, estimate_sigma

from ..core.registry import op
from ..core.volume import Volume
from ._util import slicewise, to_dtype


def _scale(img):
    info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
    return (img.astype(np.float32) / info.max, info.max) if info else (img.astype(np.float32), 1.0)


@op("preprocess.denoise_nlm")
def denoise_nlm(volume: Volume, h: float = 0.8, patch_size: int = 5,
                patch_distance: int = 6) -> Volume:
    """Non-local means per slice; ``h`` is relative to the estimated noise σ."""

    def f(s):
        a, top = _scale(s)
        sig = float(estimate_sigma(a))
        r = denoise_nl_means(a, h=h * sig, sigma=sig, patch_size=patch_size,
                             patch_distance=patch_distance, fast_mode=True)
        return to_dtype(r * top, s.dtype)

    return volume.derive(slicewise(volume, f), denoise={"method": "nlm", "h": h})


@op("preprocess.denoise_tv")
def denoise_tv(volume: Volume, weight: float = 0.05) -> Volume:
    """Total-variation (Chambolle) per slice: edge preserving, piecewise flat."""

    def f(s):
        a, top = _scale(s)
        return to_dtype(denoise_tv_chambolle(a, weight=weight) * top, s.dtype)

    return volume.derive(slicewise(volume, f), denoise={"method": "tv", "weight": weight})


@op("preprocess.gaussian")
def gaussian(volume: Volume, sigma_nm: float = 10.0) -> Volume:
    """3D Gaussian with a physical width (anisotropy-aware)."""
    sig = [sigma_nm / v for v in volume.voxel_size.as_tuple()]
    depth = (int(np.ceil(3 * sig[0])) + 1, 0, 0)
    arr = volume.data.rechunk((16, -1, -1))
    out = arr.map_overlap(lambda b: to_dtype(ndi.gaussian_filter(b.astype(np.float32), sig),
                                             b.dtype),
                          depth=depth, boundary="reflect", dtype=volume.dtype)
    return volume.derive(out, denoise={"method": "gaussian3d", "sigma_nm": sigma_nm})


@op("preprocess.median")
def median(volume: Volume, size: int = 3, mode: str = "2d") -> Volume:
    """Median filter, per slice (``2d``) or in 3D."""
    if mode == "2d":
        return volume.derive(slicewise(volume, lambda s: ndi.median_filter(s, size)))
    arr = volume.data.rechunk((16, -1, -1))
    out = arr.map_overlap(lambda b: ndi.median_filter(b, size), depth=(size // 2 + 1, 0, 0),
                          boundary="reflect", dtype=volume.dtype)
    return volume.derive(out)


@op("preprocess.unsharp")
def unsharp(volume: Volume, sigma_px: float = 2.0, amount: float = 0.6) -> Volume:
    """Unsharp-mask edge enhancement per slice."""

    def f(s):
        a = s.astype(np.float32)
        return to_dtype(a + amount * (a - ndi.gaussian_filter(a, sigma_px)), s.dtype)

    return volume.derive(slicewise(volume, f))


__all__ = ["denoise_nlm", "denoise_tv", "gaussian", "median", "unsharp"]
