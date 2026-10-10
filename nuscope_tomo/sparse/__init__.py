"""Sparse, context-driven acquisition (after Kalinin, Ziatdinov et al.).

Measure an expensive signal (EDS) at a small, actively chosen subset of
positions and reconstruct it everywhere from cheap structural context (SE
image patches) with deep kernel learning.
"""
from .dkl import DKLRegressor, extract_patches  # noqa: F401
from .active import active_sparse, reconstruct_interpolate  # noqa: F401
from .pansharpen import bse_response, cnmf_pansharpen, fusion_metrics  # noqa: F401
