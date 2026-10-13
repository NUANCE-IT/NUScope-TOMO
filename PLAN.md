# NUScope-TOMO — plan

Goal: an open, scriptable, modern alternative to **Thermo Fisher Avizo** for FIB-SEM
slice-and-view in materials science (import → pre-processing → AI segmentation →
quantification → materials analyses → presentation, driven by reproducible recipes), with
correlative chemistry built in.

Reference: *Avizo Software for FIB-SEM* datasheet (DTAS-15561042, 2026) and the Avizo
package extensions (XLab, pore network modelling, XFiber, XDVC).

## Feature map vs. Avizo for FIB-SEM

| Avizo capability | NUScope-TOMO | Status |
|---|---|---|
| Large datasets, EM / spectroscopy formats | `io.stack` (TIFF folders, multipage TIFF, Thermo Fisher pixel-size tags) → chunked Zarr v3 | done |
| Stack alignment for drift | `preprocess.align`: sub-pixel phase correlation, multi-baseline least squares, registered-stack detection, fiducial ROI | done |
| Curtaining / acquisition-artifact correction | `preprocess.destripe`, `flatten_shading`, `normalize_slices`, `auto_crop`, `tilt_correct` | done |
| DualBeam 3D Wizard (guided pre-processing) | **Studio FIB-SEM Wizard**: 7 editable steps, per-step progress, recorded as recipe | done |
| Multi-dataset co-registration, mismatched fields of view | EDS→SE registration with FOV mapping (`chem.register_to_se`) | done (2D per slice) |
| | General 3D rigid/affine registration of a second volume (BSE + SE, ToF-SIMS) | next |
| AI denoising (pre-trained / custom) | NLM, TV, anisotropic 3D Gaussian; **Noise2Void** self-supervised (MPS/CUDA) | done |
| AI segmentation with minimal annotation | Scribble-trained random forest (41 features); **2.5D U-Net** distilled from the forest's pseudo-labels | done |
| | pre-trained particle detector | next |
| Near-instant AI-assisted object selection | click-to-select object in Studio (region grow on the RF probability map) | next |
| Multi-class segmentation (cathode/binder, shale/pore, coatings) | multi-class RF + majority smoothing | done |
| 200+ measurements, custom measurements | 22 per-object measures, phase metrics, any pandas expression on the tables | done |
| Triple-phase boundaries | `analyze.tpb`: voxel-edge TPB length density, active/isolated | done |
| Pore network modelling (XPNM) | `analyze.network`: regions → nodes, shared faces → throats/contacts, coordination | done |
| Physical property simulation (XLab) | `analyze.transport`: diffusion/conduction, D_eff/D₀, tortuosity factor, MacMullin number | done |
| | Absolute permeability (Stokes / lattice Boltzmann) | next |
| Fibre tracing (XFiber) | centreline tracing, orientation tensor | later |
| Digital volume correlation (XDVC) | | later |
| Crack density per secondary particle (battery) | per-object containment statistics | next |
| Visualisation, animation, reports | GPU renders, cut-aways, property-coloured objects, networks, fields, turntables, HTML report | done |
| Recipes, reproducibility | YAML recipes, provenance log, Studio actions auto-recorded | done |
| — (beyond Avizo) | Correlative EDS chemical phase mapping with a forward simulator; browser UI next to the data; open formats; ground-truth test suite | done |

## Beyond Avizo

See [docs/IDEAS.md](docs/IDEAS.md). Built: data-health inspector with slice-thickness check,
uncertainty / RVE on every metric, live acquisition mode, Noise2Void, sparse-label U-Net.

## What "more modern than Avizo" means here

1. **Browser-based Studio**: runs on a lab workstation or HPC node and opens from any
   laptop. No licence dongle and nothing to install on the viewing machine.
2. **Everything is code**: every GUI action maps to a recipe step saved in the project,
   so results are reproducible and diffable in git.
3. **Out-of-core by default**: chunked Zarr v3 + Dask, so stacks larger than RAM work.
4. **Python API, notebooks, CLI**: the same engine from scripts, Jupyter or the UI.
5. **Open formats**: OME-Zarr, TIFF, CSV, STL/PLY, MP4. No proprietary project files.
6. **Validated**: each analysis has a ground-truth test (analytical geometry or forward model).

## Added after the Avizo comparison

Acquisition advisor; cohort statistics with detectability; sparse DKL and coupled-NMF
pan-sharpened chemistry (benchmarked with SE-invisible chemistry); self-driving slice-and-view
(AutoScript backend in dry-run, digital twin, shadow mode). See docs/IDEAS.md.

## Next milestones

1. Active learning in Studio (annotate where the model is least certain); model cards.
2. 3D multi-volume registration (SE/BSE, ToF-SIMS, EBSD) with mismatched fields of view.
3. Native EDS readers: Oxford `.h5oina`, Bruker `.bcf`.
4. Permeability (lattice Boltzmann) and electrochemical properties for battery cathodes.
5. Studio: click-to-select objects, measurement filters linked to 3D highlighting, report export.
6. First hardware session for the AutoScript backend: shadow-mode review, then a supervised
   run with conservative limits; calibrate the twin's time model against a real cycle.
7. Pan-sharpening on full EDS spectra (hundreds of energy channels) and on ToF-SIMS.
