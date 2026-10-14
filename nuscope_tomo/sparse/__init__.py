"""Sparse, context-driven acquisition (after Kalinin, Ziatdinov et al.).

Measure an expensive signal (EDS) at a small, actively chosen subset of
positions and reconstruct it everywhere from cheap structural context (SE
image patches) with deep kernel learning.
"""
from .active import active_sparse, reconstruct_interpolate  # noqa: F401
from .pansharpen import bse_response, cnmf_pansharpen, fusion_metrics  # noqa: F401


def __getattr__(name):
    # deep kernel learning needs PyTorch (pip install "nuscope-tomo[ai]"); load it only when asked for
    if name in ("DKLRegressor", "extract_patches"):
        from . import dkl
        return getattr(dkl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
