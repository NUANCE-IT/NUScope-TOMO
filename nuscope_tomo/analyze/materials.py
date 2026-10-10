"""Materials-science analyses of the kind Avizo provides as extensions:

* ``triple_phase_boundary`` — TPB length density (SOFC electrodes, catalysts)
  with active / isolated classification;
* ``transport`` — steady-state diffusion/conduction through one phase
  (finite-volume Laplace solve, XLab-style): effective diffusivity ratio,
  tortuosity factor and MacMullin number;
* ``network_model`` — pore- (or particle-) network extraction: watershed
  regions become nodes, shared faces become throats / contacts, with sizes
  and coordination numbers.

All lengths in nm unless the column says otherwise; anisotropic voxels are
handled throughout.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import cg

from ..core.registry import op
from ..core.volume import Volume


# --------------------------------------------------------------------- TPB
def _active(mask: np.ndarray, mode: str) -> np.ndarray:
    """Voxels of a phase that belong to its 'active' network."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    if mode == "largest":
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        return lab == int(np.argmax(sizes))
    # "boundary": clusters touching any face of the volume (connected to the outside)
    faces = set()
    for ax in range(3):
        for idx in (0, -1):
            faces |= set(np.unique(np.take(lab, idx, axis=ax)).tolist())
    faces.discard(0)
    return np.isin(lab, list(faces))


def triple_phase_boundary(lab: np.ndarray, spacing_nm, phases: Sequence[int],
                          active: str = "boundary") -> tuple[dict, np.ndarray]:
    """Length of lines where three phases meet.

    Voxel-edge method: every grid line parallel to an axis is shared by
    four voxels; if those contain all three phases, that segment (one voxel
    long along the axis) is TPB. A segment is *active* when each of the
    three phases touching it belongs to that phase's connected network
    (``active``: "boundary" = connected to the volume faces, "largest").
    """
    a, b, c = phases
    sp = np.asarray(spacing_nm, float)
    act = {p: _active(lab == p, active) for p in phases}
    total = active_len = 0.0
    tpb_map = np.zeros(lab.shape, bool)
    for k in range(3):
        o = [i for i in range(3) if i != k]
        sl = []
        for dy in (0, 1):
            for dx in (0, 1):
                s = [slice(None)] * 3
                s[o[0]] = slice(dy, lab.shape[o[0]] - 1 + dy)
                s[o[1]] = slice(dx, lab.shape[o[1]] - 1 + dx)
                sl.append(tuple(s))
        quad = [lab[s] for s in sl]
        has = {p: np.logical_or.reduce([q == p for q in quad]) for p in phases}
        edge = has[a] & has[b] & has[c]
        has_act = {p: np.logical_or.reduce([(q == p) & act[p][s] for q, s in zip(quad, sl)])
                   for p in phases}
        edge_act = edge & has_act[a] & has_act[b] & has_act[c]
        total += edge.sum() * sp[k]
        active_len += edge_act.sum() * sp[k]
        tpb_map[sl[0]] |= edge
    vol_um3 = lab.size * np.prod(sp) / 1e9
    out = {"tpb_length_um": total / 1e3, "tpb_density_um-2": total / 1e3 / vol_um3,
           "active_length_um": active_len / 1e3,
           "active_density_um-2": active_len / 1e3 / vol_um3,
           "active_fraction": active_len / total if total else np.nan}
    return out, tpb_map


@op("analyze.tpb", inputs=("labels",))
def tpb(labels: Volume, phases: Sequence[int] = (1, 2, 3), active: str = "boundary",
        max_voxels: int = 64_000_000) -> dict:
    """Triple-phase boundary length density (total and active)."""
    if np.prod(labels.shape) > max_voxels:
        labels = labels.preview(max_voxels)
    stats, tmap = triple_phase_boundary(labels.numpy(), labels.voxel_size.as_tuple(), phases,
                                        active)
    names = labels.meta.get("classes", {})
    stats["phases"] = " / ".join(names.get(str(p), str(p)) for p in phases)
    return {"table": pd.DataFrame([stats]),
            "map": labels.derive(tmap.astype(np.uint8), name="tpb", kind="labels",
                                 classes={"1": "TPB"})}


# --------------------------------------------------------------- transport
def effective_transport(mask: np.ndarray, spacing_nm, axis: int = 0, tol: float = 1e-8,
                        maxiter: int = 20000, return_field: bool = False) -> dict:
    """Steady diffusion/conduction through ``mask`` along ``axis``.

    Finite-volume Laplace solve with unit intrinsic diffusivity inside the
    phase and zero outside; u = 1 on the inlet slab, u = 0 on the outlet
    slab, no flux on the other faces. Returns the effective diffusivity
    ratio D_eff/D0, tortuosity factor τ = ε / (D_eff/D0) (TauFactor / XLab
    convention) and MacMullin number N_M = D0/D_eff.
    """
    sp = np.asarray(spacing_nm, float)
    eps = float(mask.mean())
    # only the cluster(s) connecting inlet and outlet carry flux
    lab, _ = ndi.label(mask)
    ids = (set(np.unique(np.take(lab, 0, axis=axis))) &
           set(np.unique(np.take(lab, -1, axis=axis)))) - {0}
    res = {"axis": "zyx"[axis], "porosity": eps, "percolating_fraction": 0.0,
           "D_eff_ratio": 0.0, "tortuosity_factor": np.inf, "macmullin": np.inf,
           "iterations": 0, "converged": True}
    if not ids:
        return res
    m = np.isin(lab, list(ids))
    res["percolating_fraction"] = float(m.sum() / max(mask.sum(), 1))
    n = mask.shape[axis]
    idx = -np.ones(mask.shape, np.int64)
    pos = np.moveaxis(np.indices(mask.shape), 0, -1)[..., axis]
    unknown = m & (pos > 0) & (pos < n - 1)
    idx[unknown] = np.arange(unknown.sum())
    N = int(unknown.sum())
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    diag = np.zeros(N)
    for k in range(3):
        g = np.prod(np.delete(sp, k)) / sp[k]  # face area / distance
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[k], s1[k] = slice(0, -1), slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        both = m[s0] & m[s1]
        i0, i1 = idx[s0][both], idx[s1][both]
        p0, p1 = pos[s0][both], pos[s1][both]
        for a_i, b_i, b_p in ((i0, i1, p1), (i1, i0, p0)):
            sel = a_i >= 0
            np.add.at(diag, a_i[sel], g)
            off = sel & (b_i >= 0)
            rows.append(a_i[off]); cols.append(b_i[off]); vals.append(np.full(off.sum(), -g))
            inlet = sel & (b_i < 0) & (b_p == 0)  # neighbour fixed at u = 1
            np.add.at(rhs, a_i[inlet], g)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N)) + sparse.diags(diag)
    it = [0]

    def cb(_):
        it[0] += 1

    x0 = 1.0 - pos[unknown] / (n - 1)  # linear profile: a good start
    try:  # algebraic multigrid preconditioner: ~10-50x fewer iterations on big grids
        import pyamg
        M = pyamg.smoothed_aggregation_solver(A.tocsr(), symmetry="symmetric").aspreconditioner()
    except ImportError:
        M = sparse.diags(1.0 / np.maximum(diag, 1e-30))
    u, info = cg(A, rhs, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=cb)
    full = np.zeros(mask.shape)
    full[unknown] = u
    full[m & (pos == 0)] = 1.0
    # flux through the first interior face (slab 0 → slab 1)
    k = axis
    g = np.prod(np.delete(sp, k)) / sp[k]
    a0 = np.take(full, 0, axis=k)
    a1 = np.take(full, 1, axis=k)
    conn = np.take(m, 0, axis=k) & np.take(m, 1, axis=k)
    Q = float(g * (a0 - a1)[conn].sum())
    L = (n - 1) * sp[k]
    area = np.prod(np.delete(np.array(mask.shape) * sp, k))
    d_ratio = Q * L / area
    res.update(D_eff_ratio=d_ratio, tortuosity_factor=eps / d_ratio if d_ratio else np.inf,
               macmullin=1 / d_ratio if d_ratio else np.inf, iterations=it[0],
               converged=info == 0)
    if return_field:
        res["field"] = np.where(m, full, np.nan).astype(np.float32)
    return res


@op("analyze.transport", inputs=("labels",))
def transport(labels: Volume, phases: Optional[Sequence[int]] = None,
              axes: Sequence[int] = (0, 1, 2), max_voxels: int = 8_000_000) -> pd.DataFrame:
    """Effective diffusivity / conductivity and tortuosity factor per phase and axis."""
    pv = labels.preview(max_voxels, isotropic=True) if np.prod(labels.shape) > max_voxels \
        else labels
    lab = pv.numpy()
    names = labels.meta.get("classes", {})
    phases = phases or [int(c) for c in np.unique(lab) if c != 0]
    rows = []
    for p in phases:
        for ax in axes:
            r = effective_transport(lab == p, pv.voxel_size.as_tuple(), ax)
            r.update(phase=p, name=names.get(str(p), str(p)),
                     grid=f"{lab.shape[0]}x{lab.shape[1]}x{lab.shape[2]}",
                     voxel_nm=round(float(pv.voxel_size.x), 1))
            rows.append(r)
    cols = ["phase", "name", "axis", "porosity", "percolating_fraction", "D_eff_ratio",
            "tortuosity_factor", "macmullin", "iterations", "converged", "grid", "voxel_nm"]
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------- network model
def extract_network(regions: np.ndarray, spacing_nm, mask: Optional[np.ndarray] = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes (regions) and links (shared faces) from a labelled region map.

    Node: volume, equivalent and inscribed diameter, centroid, coordination.
    Link: contact area, inscribed (throat) diameter, centre-to-centre length.
    """
    sp = np.asarray(spacing_nm, float)
    m = regions > 0 if mask is None else mask
    dist = ndi.distance_transform_edt(m, sampling=tuple(sp))
    ids = np.arange(1, int(regions.max()) + 1)
    vox = np.bincount(regions.ravel(), minlength=len(ids) + 1)[1:]
    cen = np.array(ndi.center_of_mass(np.ones_like(regions), regions, ids)) * sp
    rmax = ndi.maximum(dist, regions, ids)
    nodes = pd.DataFrame({"id": ids, "voxels": vox, "volume_um3": vox * np.prod(sp) / 1e9,
                          "eq_diameter_nm": (6 * vox * np.prod(sp) / np.pi) ** (1 / 3),
                          "inscribed_diameter_nm": 2 * rmax,
                          "z_um": cen[:, 0] / 1e3, "y_um": cen[:, 1] / 1e3,
                          "x_um": cen[:, 2] / 1e3})
    pairs, areas, throat_r = [], [], []
    for k in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[k], s1[k] = slice(0, -1), slice(1, None)
        a, b = regions[tuple(s0)], regions[tuple(s1)]
        sel = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel])
        d = np.minimum(dist[tuple(s0)][sel], dist[tuple(s1)][sel])
        pairs.append(lo.astype(np.int64) * (len(ids) + 1) + hi)
        areas.append(np.full(sel.sum(), np.prod(np.delete(sp, k))))
        throat_r.append(d)
    key = np.concatenate(pairs)
    if key.size == 0:
        nodes["coordination"] = 0
        return nodes, pd.DataFrame(columns=["a", "b", "area_um2", "throat_diameter_nm",
                                            "length_nm"])
    area = np.concatenate(areas)
    tr = np.concatenate(throat_r)
    uk, inv = np.unique(key, return_inverse=True)
    a_id, b_id = uk // (len(ids) + 1), uk % (len(ids) + 1)
    links = pd.DataFrame({"a": a_id, "b": b_id,
                          "area_um2": np.bincount(inv, area) / 1e6,
                          "throat_diameter_nm": 2 * ndi.maximum(tr, inv, np.arange(len(uk)))})
    ca, cb = cen[a_id - 1], cen[b_id - 1]
    links["length_nm"] = np.linalg.norm(ca - cb, axis=1)
    coord = np.bincount(np.concatenate([a_id, b_id]), minlength=len(ids) + 1)[1:]
    nodes["coordination"] = coord
    return nodes, links


@op("analyze.network", inputs=("labels",))
def network(labels: Volume, min_contact_um2: float = 0.0) -> dict:
    """Pore / particle network model from separated regions (nodes + throats/contacts)."""
    nodes, links = extract_network(labels.numpy(), labels.voxel_size.as_tuple())
    if min_contact_um2:
        links = links[links.area_um2 >= min_contact_um2].reset_index(drop=True)
        coord = np.bincount(np.concatenate([links.a, links.b]),
                            minlength=len(nodes) + 1)[1:]
        nodes["coordination"] = coord
    return {"nodes": nodes, "links": links}


__all__ = ["triple_phase_boundary", "tpb", "effective_transport", "transport",
           "extract_network", "network"]
