"""Build the results gallery for the fib_sem_335 recipe run.

    python examples/showcase.py [project_dir]

Reads the volumes/tables written by ``nuscope run examples/fib_sem_335.yaml``
and writes figures, renders and animations into ``<project>/gallery``.
The EDS panel uses the *simulated* EDS forward model on the real
segmentation (no EDS was acquired for this dataset) and is labelled as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nuscope_tomo import Volume  # noqa: E402
from nuscope_tomo.chem import (chemical_phases, composition, composition_by_label,  # noqa: E402
                               denoise_counts, element_rgb, register_to_se, simulate_eds)
from nuscope_tomo.preprocess import auto_crop  # noqa: E402
from nuscope_tomo.viz import PALETTE, render_objects, render_phases, render_volume, turntable  # noqa: E402
from nuscope_tomo.viz.figures import STYLE, compare, label_cmap, ortho_views  # noqa: E402

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "projects/fib_sem_335/recipe_run")
OUT = PROJ / "gallery"
OUT.mkdir(parents=True, exist_ok=True)
V = PROJ / "volumes"
T = PROJ / "tables"

raw = Volume.from_zarr("projects/fib_sem_335/raw.zarr")
crop = auto_crop(raw)
corrected = Volume.from_zarr(V / "corrected.zarr")
den = Volume.from_zarr(V / "denoised.zarr")
phases = Volume.from_zarr(V / "phases.zarr")
particles = Volume.from_zarr(V / "particles.zarr")
names = phases.meta.get("classes", {})
print(den, phases, particles, sep="\n")


def step(msg):
    print(f"· {msg}", flush=True)


# 1 ── raw frame with detected ROI ------------------------------------------
step("raw frame + ROI")
c = crop.meta["crop"]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(raw.get_slice(0, 400), cmap="gray")
    ax.add_patch(plt.Rectangle((c["x"][0], c["y"][0]), c["x"][1] - c["x"][0],
                               c["y"][1] - c["y"][0], fill=False, ec="#E69F00", lw=2))
    ax.text(c["x"][0] + 10, c["y"][0] + 40, "auto-detected block face", color="#E69F00",
            fontsize=12, fontweight="bold")
    ax.text(900, 110, "Pt cap", color="w", fontsize=11)
    ax.text(30, 800, "trench", color="w", fontsize=11, rotation=90)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Raw SE slice 400 / 812", loc="left")
    fig.savefig(OUT / "01_raw_roi.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 2 ── pre-processing before/after (XY zoom + XZ reslice) --------------------
step("pre-processing comparison")
z = 400
imgs = {"cropped raw": crop.get_slice(0, z),
        "destriped + shading + normalised": corrected.get_slice(0, z),
        "binned 2× + non-local means": np.kron(den.get_slice(0, z), np.ones((2, 2)))[
            : corrected.shape[1], : corrected.shape[2]]}
compare(imgs, out=OUT / "02_preprocess_xy.png", clim=(60, 190), zoom=((300, 560), (560, 940)),
        title="Pre-processing chain (XY, slice 400)")
plt.close("all")
xz = {"cropped raw": crop.get_slice(1, 560)[:300].T,
      "corrected": corrected.get_slice(1, 560)[:300].T}
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for a, (k, im) in zip(axes, xz.items()):
        # rows = x (5 nm), cols = z (50 nm): true physical aspect
        a.imshow(im, cmap="gray", vmin=60, vmax=190, aspect=5 / 50, interpolation="lanczos")
        a.set_title(f"XZ reslice, {k}  —  Z (milling) →", loc="left")
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "03_preprocess_xz.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 3 ── drift diagnostics -------------------------------------------------------
step("drift")
drift = pd.read_csv(T / "align.drift.csv")
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].plot(drift.z, drift.dy_step, lw=.7, color=PALETTE[1], label="dy")
    ax[0].plot(drift.z, drift.dx_step, lw=.7, color=PALETTE[0], label="dx")
    ax[0].set_title("Slice-to-slice shift (px) — noise level → stack already registered", loc="left")
    ax[0].set_xlabel("slice"), ax[0].legend(frameon=False)
    ax[1].hist(drift[["dy_step", "dx_step"]].to_numpy().ravel(), 60, color=PALETTE[2])
    ax[1].set_title("Step distribution (px)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "04_drift.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 4 ── ortho views with segmentation --------------------------------------------
step("ortho views")
ortho_views(den.roi(slice(250, 550)), phases.roi(slice(250, 550)), pos=(150, 250, 340),
            out=OUT / "05_ortho_phases.png",
            title="Random-forest phase segmentation  —  " +
            "  ".join(f"■ {names.get(str(k), k)}" for k in range(1, 5)), alpha=0.5)
plt.close("all")

# 5 ── phase statistics ------------------------------------------------------------
step("phase stats")
ph = pd.read_csv(T / "phase_metrics.phases.csv")
sd = pd.read_csv(T / "phase_metrics.size_distribution.csv")
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    cols = [PALETTE[(int(c) - 1) % 10] for c in ph["class"]]
    ax[0].bar(ph["name"], ph["volume_fraction"] * 100, color=cols)
    ax[0].set_ylabel("volume %"), ax[0].set_title("Phase volume fractions", loc="left")
    ax[1].bar(ph["name"], ph["specific_surface_um-1"], color=cols)
    ax[1].set_ylabel("µm⁻¹"), ax[1].set_title("Specific surface area S/V", loc="left")
    for (c, n), g in sd.groupby(["class", "name"]):
        ax[2].plot(g["diameter_nm"], g["cumulative_fraction"], "o-", ms=3,
                   color=PALETTE[(int(c) - 1) % 10], label=n)
    ax[2].set_xscale("log"), ax[2].set_xlabel("diameter (nm)")
    ax[2].set_title("Continuous size distribution (fraction ≥ d)", loc="left")
    ax[2].legend(frameon=False, fontsize=9)
    for a in ax[:2]:
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "06_phase_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 6 ── particle statistics ------------------------------------------------------
step("particle stats")
pm = pd.read_csv(T / "particles_measured.csv")
inner = pm[~pm.touches_border]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].hist(inner.eq_diameter_nm, 40, color=PALETTE[0])
    ax[0].set_title(f"Equivalent diameter (n = {len(inner)})", loc="left"), ax[0].set_xlabel("nm")
    ax[1].hist(inner.sphericity.clip(0, 1), 40, color=PALETTE[2])
    ax[1].set_title("Sphericity", loc="left")
    sc = ax[2].scatter(inner.volume_um3, inner.sphericity, s=6, c=inner.elongation, cmap="viridis")
    ax[2].set_xscale("log"), ax[2].set_xlabel("volume (µm³)"), ax[2].set_ylabel("sphericity")
    ax[2].set_title("Shape vs size (colour = elongation)", loc="left")
    plt.colorbar(sc, ax=ax[2], fraction=0.04)
    big = inner[inner.voxels > 200]
    ax[3].hist(big.theta_deg, 18, range=(0, 90), color=PALETTE[4])
    ax[3].set_title("Major-axis angle to milling axis Z", loc="left"), ax[3].set_xlabel("deg")
    fig.tight_layout()
    fig.savefig(OUT / "07_particle_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# 7 ── 3D renders -----------------------------------------------------------------
step("3D renders")
sub = (slice(200, 600), slice(None), slice(None))   # a 20 µm long block for clarity
den_r = den.roi(*sub).preview(max_voxels=200**3, isotropic=True)
ph_r = phases.roi(*sub).preview(max_voxels=200**3, isotropic=True)
cam = {"zoom": 1.35}
render_volume(den_r, out=str(OUT / "08_volume_render.png"), cmap="bone",
              title="Direct volume rendering (denoised SE)", camera=cam)
render_phases(ph_r, den_r, classes=[1, 2], out=str(OUT / "09_phases_3d.png"),
              title="Dark phase + pores, cut-away with SE data", camera=cam)
p_sub = particles.roi(*sub)
pm_map = dict(zip(pm.label, pm.eq_diameter_nm))
p_r = p_sub.preview(max_voxels=260**3, isotropic=True)
render_objects(p_r, {k: pm_map.get(k, np.nan) for k in np.unique(p_r.numpy()) if k},
               cmap="turbo", label="eq. diameter (nm)", out=str(OUT / "10_particles_3d.png"),
               title="Separated dark-phase particles coloured by size", camera=cam)
sph_map = dict(zip(pm.label, pm.sphericity))
render_objects(p_r, {k: sph_map.get(k, np.nan) for k in np.unique(p_r.numpy()) if k},
               cmap="viridis", clim=(0.4, 1.0), label="sphericity",
               out=str(OUT / "11_particles_sphericity.png"),
               title="Particles coloured by sphericity", camera=cam)

step("turntable")
turntable(lambda: render_phases(ph_r, den_r, classes=[1, 2], out=None, size=(1280, 960),
                                camera=cam, legend=False),
          str(OUT / "turntable_phases.mp4"), n_frames=240, fps=30)

# 8 ── correlative EDS (simulated forward model on the real segmentation) -------------
step("EDS (simulated)")
zs = slice(300, 420)
ph_block = phases.roi(zs).numpy()
se_block = den.roi(zs)
comp = {1: {"C": 0.7, "O": 0.2, "Ni": 0.05, "Mn": 0.05},      # pore (epoxy/void)
        2: {"Ni": 0.15, "Mn": 0.55, "Co": 0.05, "O": 0.25},   # dark phase: Mn-rich
        3: {"Ni": 0.45, "Mn": 0.15, "Co": 0.15, "O": 0.25},   # grey phase
        4: {"Ni": 0.60, "Mn": 0.05, "Co": 0.10, "O": 0.25}}   # bright phase: Ni-rich
eds, truth = simulate_eds(ph_block, comp, se_block.voxel_size, bin_xy=6, every=4,
                          counts_per_pixel=40, shift_px=(1.4, -0.9), psf_nm=60)
reg = register_to_se(eds, se_block)
cmp_ = composition(denoise_counts(reg, 40))
chem_lab, chem_tab = chemical_phases(cmp_, 3, method="gmm")
table = composition_by_label(cmp_, ph_block, names)
table.to_csv(OUT / "eds_composition_by_phase.csv", index=False)
chem_tab.to_csv(OUT / "eds_chemical_phases.csv", index=False)
k = len(reg.z_index) // 2
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6))
    ax[0].imshow(se_block.get_slice(0, int(reg.z_index[k])), cmap="gray")
    ax[0].set_title("SE (denoised)", loc="left")
    ax[1].imshow(eds.maps["Mn"][k], cmap="magma")
    ax[1].set_title(f"raw Mn counts ({eds.shape[1]}×{eds.shape[2]} px)", loc="left")
    ax[2].imshow(element_rgb(cmp_, ["Mn", "Ni", "O"], k))
    ax[2].set_title("registered composition  R=Mn G=Ni B=O", loc="left")
    ax[3].imshow(chem_lab[k], cmap=label_cmap(3, 1.0), vmin=0, vmax=3, interpolation="nearest")
    ax[3].set_title("unsupervised chemical phases (GMM)", loc="left")
    els = ["Mn", "Ni", "Co", "O", "C"]
    x = np.arange(len(table))
    for i, e in enumerate(els):
        ax[4].bar(x + i * 0.16 - 0.32, table[f"{e}_mean"], 0.16, label=e, color=PALETTE[i + 4])
    ax[4].set_xticks(x, table["name"], rotation=15)
    ax[4].set_title("composition per SE phase", loc="left"), ax[4].legend(frameon=False, ncol=5, fontsize=9)
    for a in ax[:4]:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Correlative EDS tomography — SIMULATED EDS on the real segmentation "
                 f"(true offset {truth['shift_px']} px, recovered "
                 "({:.2f}, {:.2f}) px)".format(*(-np.median(reg.shifts_px, 0))),
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "12_eds_simulated.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

print("gallery →", OUT)
