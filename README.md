# NuScope-TOMO

**Open FIB-SEM slice-and-view tomography — from raw slices to quantified, chemically
resolved 3D microstructure.**

Created by **Roberto dos Reis ([@rmsreis](https://github.com/rmsreis))** · NUANCE Center,
Northwestern University · [NUANCE-IT/NUScope-TOMO](https://github.com/NUANCE-IT/NUScope-TOMO)

NuScope-TOMO covers the Thermo Fisher Amira FIB-SEM workflow (import →
pre-processing → AI segmentation → quantification → visualization/animation →
recipes). It adds correlative EDS chemistry, and everything is available as code, as
reproducible YAML recipes and in a browser-based UI.

<p>
  <img src="docs/images/09_phases_3d.jpg" width="49%" alt="Phase surfaces with cut-away">
  <img src="docs/images/10_particles_3d.jpg" width="49%" alt="Particles coloured by size">
</p>

*FIB-SEM sample 335 (812 slices) processed by `examples/fib_sem_335.yaml` in under 5 minutes:
dark phase with a grey-value cut-away (left) and 1 934 separated particles coloured by equivalent
diameter (right).*

| Stage | What you get |
|---|---|
| **Import** | TIFF/PNG folders (natural sort), multipage TIFF, Thermo Fisher pixel-size tags, streamed into chunked Zarr v3, so stacks larger than RAM are fine |
| **Pre-processing** | block-face auto-crop (Pt cap / trench / padding), multi-baseline drift solve with already-registered detection, wavelet–FFT curtain removal, shading flattening, slice brightness normalisation, NLM / TV / anisotropic 3D denoising, binning, isotropic resampling, 52° tilt calibration |
| **Segmentation** | Otsu / multi-Otsu / hysteresis, **random-forest classifier trained from a few scribbles** (41 multi-scale features), majority smoothing, watershed separation of touching particles |
| **Quantification** | per object: volume, surface area, equivalent diameter, sphericity, solidity, Feret, ellipsoid axes, elongation/flatness, orientation, nearest neighbour, intensity/element means. Per phase: volume fraction, specific surface, interface density, Euler number, percolation, geodesic tortuosity, continuous size distribution |
| **Chemistry (EDS)** | element-map stacks → registration to SE → 3D composition → unsupervised chemical phases (k-means / GMM / NMF) → composition per SE phase or object; forward simulator for validation |
| **Visualization** | GPU renders (volume, phase surfaces with cut-away, objects coloured by any measurement), turntable MP4s, figure panels with true aspect and scale bars |
| **Studio** | browser UI: linked XY/XZ/YZ views, WebGL2 ray-marched 3D (DVR / MIP / iso / labels, clipping), scribble annotation → train → segment, every operation as a form, sortable measurement tables. All actions are auto-recorded as a recipe |
| **Recipes** | YAML pipelines, `nuscope run`; every step logged with parameters, timings and library versions in `provenance.json` |

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Use

```bash
nuscope run examples/fib_sem_335.yaml          # full pipeline on a slice-and-view stack
nuscope studio projects/fib_sem_335            # open http://127.0.0.1:8765
nuscope ops                                    # list operations and parameters
python examples/showcase.py                    # figures, renders and animation
```

From Python:

```python
from nuscope_tomo import Volume
from nuscope_tomo.preprocess import auto_crop, align, destripe, denoise_nlm
from nuscope_tomo.segment import random_forest
from nuscope_tomo.analyze import measure_objects

vol = Volume.from_zarr("raw.zarr")
vol = denoise_nlm(destripe(align(auto_crop(vol))["volume"]))
phases = random_forest(vol, scribbles="annotations.json", classes=3)["volume"]
```

Remote use: run `nuscope studio --host 0.0.0.0` on the workstation that holds the data
and open it from any laptop. The browser only receives slices and a downsampled 3D preview.

## Design notes

- **Out-of-core first.** Volumes are Dask arrays over Zarr; slice-wise operations stream.
  Connected components, watershed and surface meshing run in memory, on a binned volume
  or an ROI for very large data.
- **Anisotropy everywhere.** FIB-SEM voxels are typically 5–10 nm in XY and 10–50 nm in
  Z. Distances, filters, surface areas and renders all use physical spacing.
- **Validated against ground truth.** `pytest` runs 12 tests on synthetic phantoms.
  They cover drift recovery, stripe removal, sphere volume, area and sphericity (within 1–2 %),
  percolation and tortuosity, random-forest accuracy, watershed splitting, EDS registration
  and phase recovery, the recipe engine and the Studio API.
- **Honest alignment.** Correlation cannot separate stage drift from microstructure that is
  inclined to the milling axis, which is especially a problem with thick slices. `align`
  therefore detects stacks that are already registered and leaves them alone, and supports
  fiducial ROIs and a high-pass drift model.

## Project layout

```
nuscope_tomo/
  core/        Volume, operation registry, recipes + provenance
  io/          slice import/export
  preprocess/  geometry, alignment, artifact correction, denoising
  segment/     thresholds, random-forest classifier, object separation
  analyze/     object morphometry, phase metrics
  chem/        EDS chemical tomography (+ simulator)
  viz/         GPU rendering, animation, figures
  studio/      FastAPI server + WebGL2 front-end
  cli.py       `nuscope` command
examples/      recipe for the sample dataset, showcase script
tests/         ground-truth tests
```

See [PLAN.md](PLAN.md) for the roadmap: deep-learning segmentation, Noise2Void,
native `.h5oina` / `.bcf` EDS readers.

## Citation

If NuScope-TOMO helps your research, please cite it using [CITATION.cff](CITATION.cff).
Author: Roberto dos Reis (@rmsreis).
