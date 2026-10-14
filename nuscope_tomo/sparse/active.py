"""Active sparse acquisition loop.

Start from a small random set of measured positions, fit the DKL model on
(SE context → measured composition), then repeatedly measure the batch of
positions with the largest posterior variance (spread out by a minimum
distance) until the budget is spent. Returns reconstructed maps everywhere.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import griddata



def _spread_top(var: np.ndarray, coords: np.ndarray, taken: np.ndarray, k: int,
                min_dist: float) -> np.ndarray:
    order = np.argsort(-var)
    chosen: list[int] = []
    for i in order:
        if taken[i]:
            continue
        if chosen and np.min(np.hypot(*(coords[chosen] - coords[i]).T)) < min_dist:
            continue
        chosen.append(i)
        if len(chosen) == k:
            break
    return np.array(chosen, int)


def active_sparse(se_img: np.ndarray, grid_shape: tuple[int, int], bin_xy: int,
                  measure: Callable[[np.ndarray], np.ndarray], budget: float = 0.05,
                  init: float = 0.01, rounds: int = 4, strategy: str = "variance",
                  patch: int = 24, min_dist: float = 2.0, seed: int = 0, explore: float = 0.3,
                  log: Callable[[str], None] | None = None) -> dict:
    """Sparse EDS on one slice.

    ``grid_shape`` is the EDS pixel grid; EDS pixel (i, j) covers SE pixels
    [i*bin, (i+1)*bin) × [j*bin, (j+1)*bin). ``measure(idx)`` returns the
    measured values (N, E) at flat EDS indices ``idx``.
    """
    rng = np.random.default_rng(seed)
    ny, nx = grid_shape
    coords = np.argwhere(np.ones(grid_shape, bool))
    centers = coords * bin_xy + bin_xy // 2
    from .dkl import DKLRegressor, extract_patches   # PyTorch: the optional [ai] extra
    P = extract_patches(se_img, centers, patch)
    N = len(coords)
    n_init = max(8, int(init * N))
    n_total = max(n_init, int(budget * N))
    taken = np.zeros(N, bool)
    idx = np.sort(rng.choice(N, n_init, replace=False))
    taken[idx] = True
    Y = measure(idx)
    per_round = max(1, (n_total - n_init) // max(rounds, 1))
    history = []
    model = None
    model = DKLRegressor(seed=seed)
    for r in range(rounds + 1):
        meas = np.flatnonzero(taken)
        model.fit(P[meas], Y)
        if r == rounds or taken.sum() >= n_total:
            break
        if strategy == "variance":
            # mostly where the model is least sure, partly random for coverage (pure
            # variance sampling chases rare outliers such as pores)
            _, var = model.predict(P)
            var[taken] = -np.inf
            n_var = int(round(per_round * (1 - explore)))
            new = _spread_top(var, coords, taken, n_var, min_dist)
            pool = np.setdiff1d(np.flatnonzero(~taken), new)
            new = np.concatenate([new, rng.choice(pool, per_round - len(new), replace=False)])
        else:
            new = rng.choice(np.flatnonzero(~taken), per_round, replace=False)
        new = np.sort(new)
        Ynew = measure(new)
        # keep Y aligned with np.flatnonzero(taken) order
        allidx = np.concatenate([meas, new])
        Yall = np.concatenate([Y, Ynew])
        taken[new] = True
        order = np.argsort(allidx)
        Y = Yall[order]
        history.append({"round": r, "measured": int(taken.sum())})
        if log:
            log(f"sparse: round {r} measured {taken.sum()} / {N}")
    mu, var = model.predict(P)
    meas = np.flatnonzero(taken)
    return {"mean": mu.reshape(ny, nx, -1), "variance": var.reshape(ny, nx),
            "measured_mask": taken.reshape(ny, nx), "fraction_measured": taken.mean(),
            "measured_idx": meas, "measured_values": Y, "history": history}


def reconstruct_interpolate(grid_shape, idx: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Context-free baseline: linear interpolation of sparse measurements."""
    ny, nx = grid_shape
    pts = np.column_stack(np.unravel_index(idx, grid_shape))
    gy, gx = np.mgrid[:ny, :nx]
    out = np.stack([griddata(pts, Y[:, e], (gy, gx), method="linear") for e in range(Y.shape[1])],
                   -1)
    near = np.stack([griddata(pts, Y[:, e], (gy, gx), method="nearest")
                     for e in range(Y.shape[1])], -1)
    return np.where(np.isfinite(out), out, near)
