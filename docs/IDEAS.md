# Features Avizo does not have

Brainstorm for NuScope-TOMO, created by Roberto dos Reis (@rmsreis). Each idea names the
gap in the Avizo FIB-SEM workflow, what NuScope would do instead, and its status.
Ordered by how much they change what a microscopist can claim from a dataset.

## 1. Trust in the numbers

| Idea | Gap in Avizo | NuScope | Status |
|---|---|---|---|
| **Uncertainty on every measurement** | Measurements are single numbers | Block-bootstrap sampling error plus segmentation-boundary sensitivity (±1 voxel), reported as confidence intervals on volume fraction, S/V, tortuosity | building |
| **Representative-volume check (RVE)** | No answer to "is my FIB volume big enough?" | Metric vs. sub-volume size; reports the size at which each metric is stable to a tolerance, and how many RVEs the volume holds | building |
| **Data-health inspector** | Artifacts are found by eye | Per-slice SNR, sharpness, brightness drift, curtaining index, duplicate/missing slices, drift residuals, and an automatic calibration check that flags features stretched along the milling axis (the 50 → 20 nm error on sample 335 would have been caught) | building |
| **Segmentation uncertainty map** | Hard labels only | Per-voxel class probability and entropy from the classifier; shown as an overlay; drives active learning | building |
| Ground-truth test suite | Closed validation | Every analysis has an analytical or forward-model test that runs in CI | done |

## 2. While the microscope is still milling

| Idea | Gap | NuScope | Status |
|---|---|---|---|
| **Live acquisition mode** | Analysis starts after acquisition ends | `nuscope watch`: ingests slices as Auto Slice & View writes them, pre-processes and segments on the fly, updates Studio live, and reports when metrics reach RVE so milling can stop early | building |
| Acquisition advisor | None | From early slices: recommends slice thickness and pixel size from feature sizes (Nyquist), predicts the volume needed for a target confidence | next |

## 3. AI that is open and accountable

| Idea | Gap | NuScope | Status |
|---|---|---|---|
| **Self-supervised denoising (Noise2Void)** | Pre-trained or vendor-trained models | Trains on the dataset itself, no clean target needed, Apple MPS / CUDA | building |
| **Sparse-label U-Net** | Deep learning needs dense ground truth or a vendor model | Trains from the same scribbles as the random forest (masked loss), exports the model file with the recipe | building |
| Active learning | User guesses where to annotate | Studio highlights the slices and regions where the model is least certain | next |
| Natural-language recipes | Menu-driven | "Segment pores and compute tortuosity along Z" → validated YAML recipe from the operation registry (Claude API), reviewed before it runs | next |
| Model cards | Opaque models | Each trained model saved with its training data summary, features, metrics and provenance | next |

## 4. Beyond one dataset

| Idea | Gap | NuScope | Status |
|---|---|---|---|
| Cohort comparison | One project at a time | Compare many samples / conditions with statistics (e.g. cycled vs pristine cathodes) | next |
| Statistical descriptors + synthetic twins | Only measured volume | Two-point correlation, chord lengths; generate larger statistically equivalent volumes for simulation | next |
| FAIR export | Proprietary project files | One-click RO-Crate package: data, recipe, versions, report, ready for a DOI | next |
| Remote / shared Studio | Desktop licence per seat | Browser UI next to the data; multi-user annotation and comments at 3D positions | partly done |
