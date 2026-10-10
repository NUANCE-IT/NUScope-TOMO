# NuScope-TOMO — plan

Goal: an open, scriptable, modern alternative to the Thermo Fisher Amira FIB-SEM
workflow (import → pre-processing → segmentation → quantification →
visualization/animation, driven by reproducible recipes), plus correlative
**chemical (EDS) tomography** from FIB slice-and-view.

## Feature map vs. Amira (FIB-SEM datasheet)

| Amira workflow stage | NuScope-TOMO module | Status |
|---|---|---|
| Data import (large stacks, microscopy formats) | `io.stack`, `io.metadata` (TIFF folders, multipage TIFF, TFS/FEI + AZtec tags), OME-Zarr multiscale store | v0.2 |
| Stack alignment / 3D registration | `preprocess.align` (sub-pixel phase correlation, robust drift model, outlier rejection) | v0.2 |
| Curtaining / artifact correction | `preprocess.destripe` (wavelet–FFT destriping), `preprocess.shading` (illumination flattening), tilt/shear correction | v0.2 |
| Noise reduction (AI denoise in Amira) | `preprocess.denoise` (NLM, bilateral, TV, anisotropic 3D); self-supervised Noise2Void hook | v0.2 / v0.3 |
| AI segmentation with minimal annotation | `segment.classifier` (multi-scale-feature random forest from sparse scribbles, multi-class), `segment.threshold` (Otsu, multi-Otsu, hysteresis), `segment.separate` (watershed) | v0.2 |
| 200+ measurements | `analyze.labels` (per-object 3D morphometrics), `analyze.phase` (volume fraction, specific surface, connectivity, tortuosity, local thickness / pore-size distribution, spatial statistics) | v0.2 |
| Visualization & animation | `viz.render` (GPU volume + surface rendering, ortho slices), `viz.animate` (turntable / fly-through MP4), **NuScope Studio** (browser UI) | v0.2 |
| Recipe-based automation | `recipe` (YAML pipelines, every step logged with parameters + provenance hash), `nuscope` CLI | v0.2 |
| — (beyond Amira) | **EDS chemical tomography**: element-map stacks registered to SE, 3D element volumes, unsupervised phase mapping (NMF / k-means), composition per segmented object | v0.2 |

## What "more modern than Amira" means here

1. **Browser-based Studio** — runs locally or on a lab GPU workstation/HPC node and
   is opened from any laptop; no licence dongle, no install on the viewing machine.
2. **Everything is code** — every GUI action maps to a recipe step; the recipe is
   saved in the project, so results are reproducible and diffable in git.
3. **Out-of-core by default** — chunked Zarr v3 + Dask, multiscale pyramids, so
   stacks larger than RAM work end to end.
4. **Python API, notebooks, CLI** — the same engine from scripts, Jupyter, or the UI.
5. **Open formats** — OME-Zarr, TIFF, CSV/Parquet, STL/PLY/glTF; no proprietary project format.
6. **Correlative chemistry built in**, not a separate product.

## Phases

1. Foundation — package layout, `Volume`, project + provenance, recipe engine, CLI.
2. Pre-processing — ROI auto-crop, alignment, destriping, shading, denoising, resampling.
3. Segmentation — thresholds, scribble-trained random forest, watershed separation.
4. Quantification — per-object and phase-level metrics, CSV export.
5. Chemistry — EDS stack import, registration, 3D element maps, phase clustering.
6. Visualization — publication renders, animations.
7. Studio — FastAPI backend + WebGL2 volume renderer and slice viewer.
8. Validation — tests on synthetic phantoms with known ground truth, then the
   real dataset `projects/fib_sem_335`, results gallery.

## Later (v0.3+)

- Deep-learning segmentation (U-Net, PyTorch/MPS) trained from Studio annotations.
- Noise2Void self-supervised denoising.
- Direct reading of Oxford `.h5oina` / Bruker `.bcf` EDS hypermaps.
- Multi-user annotation in Studio.
