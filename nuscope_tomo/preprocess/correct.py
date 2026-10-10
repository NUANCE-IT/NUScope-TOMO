"""Artifact correction: curtaining (destriping), shading/illumination
flattening and slice-to-slice brightness normalisation."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pywt
from scipy import ndimage as ndi

from ..core.registry import op
from ..core.volume import Volume
from ._util import sample_slices, slicewise, to_dtype


# ----------------------------------------------------------------- curtaining
def destripe_slice(img: np.ndarray, level: int = 5, sigma: float = 3.0,
                   wavelet: str = "db6", angle_deg: float = 0.0) -> np.ndarray:
    """Wavelet–FFT stripe removal (Münch et al., Opt. Express 17, 8567, 2009).

    FIB curtains run along the ion-beam direction (image Y). They are
    nearly constant along Y, so in the vertical-detail wavelet coefficients
    they collapse to the lowest Y-frequencies, which are damped with a
    Gaussian notch. ``angle_deg`` rotates first for tilted curtains.
    """
    a = img.astype(np.float32)
    if angle_deg:
        a = ndi.rotate(a, -angle_deg, reshape=True, order=1, mode="reflect")
    coeffs = pywt.wavedec2(a, wavelet, level=level, mode="symmetric")
    out = [coeffs[0]]
    for ch, cv, cd in coeffs[1:]:
        f = np.fft.rfft(cv, axis=0)
        k = np.arange(f.shape[0])[:, None]
        f *= 1.0 - np.exp(-(k ** 2) / (2.0 * sigma ** 2))
        out.append((ch, np.fft.irfft(f, n=cv.shape[0], axis=0), cd))
    r = pywt.waverec2(out, wavelet, mode="symmetric")
    if angle_deg:
        r = ndi.rotate(r, angle_deg, reshape=True, order=1, mode="reflect")
        dy, dx = (r.shape[0] - img.shape[0]) // 2, (r.shape[1] - img.shape[1]) // 2
        r = r[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return to_dtype(r[: img.shape[0], : img.shape[1]], img.dtype)


@op("preprocess.destripe")
def destripe(volume: Volume, level: int = 5, sigma: float = 3.0, wavelet: str = "db6",
             angle_deg: float = 0.0) -> Volume:
    """Remove FIB curtaining stripes (wavelet–FFT filter)."""
    f = lambda s: destripe_slice(s, level, sigma, wavelet, angle_deg)  # noqa: E731
    return volume.derive(slicewise(volume, f), destripe={"level": level, "sigma": sigma})


# -------------------------------------------------------------------- shading
def estimate_shading(volume: Volume, sigma_px: Optional[float] = None,
                     n_samples: int = 32) -> np.ndarray:
    """Smooth multiplicative illumination field (mean ≈ 1), shared by all
    slices: median over z removes structure, a wide Gaussian keeps only
    the slow detector/geometry gradient."""
    st, _ = sample_slices(volume, n_samples)
    med = np.median(st.astype(np.float32), 0)
    sigma = sigma_px or max(med.shape) / 8
    valid = med > 0
    num = ndi.gaussian_filter(np.where(valid, med, 0), sigma)
    den = ndi.gaussian_filter(valid.astype(np.float32), sigma) + 1e-6
    field = num / den
    return (field / np.mean(field[valid])).astype(np.float32)


@op("preprocess.flatten_shading")
def flatten_shading(volume: Volume, sigma_px: Optional[float] = None) -> Volume:
    """Divide out the slow illumination gradient (shading correction)."""
    field = estimate_shading(volume, sigma_px)
    f = lambda s: to_dtype(s.astype(np.float32) / field, s.dtype)  # noqa: E731
    return volume.derive(slicewise(volume, f), shading_corrected=True)


@op("preprocess.normalize_slices")
def normalize_slices(volume: Volume, window: int = 15, match: str = "percentile") -> Volume:
    """Remove slice-to-slice brightness/contrast flicker by matching each
    slice's 2nd/98th percentiles to a running median along z."""
    step = max(1, volume.shape[1] // 512)
    arr = volume.data.rechunk((8, -1, -1))
    stats = arr.map_blocks(
        lambda b: np.percentile(b[:, ::step, ::step].reshape(len(b), -1), (2, 98), axis=1
                                ).T[:, :, None].astype(np.float32),
        dtype=np.float32, chunks=(arr.chunks[0], (2,), (1,)))
    p = stats.compute()[:, :, 0]
    ref = ndi.median_filter(p, size=(window, 1), mode="nearest")
    gain = (ref[:, 1] - ref[:, 0]) / np.maximum(p[:, 1] - p[:, 0], 1e-3)
    offset = ref[:, 0] - gain * p[:, 0]
    arr = volume.data.rechunk((8, -1, -1))

    def block(b, block_info=None):
        z0 = block_info[0]["array-location"][0][0]
        return np.stack([to_dtype(s.astype(np.float32) * gain[z0 + k] + offset[z0 + k], s.dtype)
                         for k, s in enumerate(b)])

    return volume.derive(arr.map_blocks(block, dtype=volume.dtype), slice_normalized=True)


@op("preprocess.rescale_intensity")
def rescale_intensity(volume: Volume, low_pct: float = 0.5, high_pct: float = 99.5,
                      dtype: str = "uint8") -> Volume:
    """Stretch the robust intensity range to the full output dtype."""
    sub = volume.data[::4, ::4, ::4].compute()
    nz = sub[sub > 0] if (sub > 0).mean() > 0.5 else sub.ravel()
    lo, hi = np.percentile(nz, (low_pct, high_pct))
    top = np.iinfo(dtype).max if np.issubdtype(np.dtype(dtype), np.integer) else 1.0
    scale = top / max(hi - lo, 1e-6)
    out = ((volume.data.astype(np.float32) - lo) * scale).clip(0, top).astype(dtype)
    return volume.derive(out, intensity_window=[float(lo), float(hi)])


__all__ = ["destripe_slice", "destripe", "estimate_shading", "flatten_shading",
           "normalize_slices", "rescale_intensity"]
