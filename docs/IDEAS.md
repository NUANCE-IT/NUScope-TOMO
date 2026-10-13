# Features Avizo does not have

Brainstorm for NUScope-TOMO, created by Roberto dos Reis (@rmsreis). Each idea names the
gap in the Avizo FIB-SEM workflow, what NUScope would do instead, and its status.
Ordered by how much they change what a microscopist can claim from a dataset.

## 1. Trust in the numbers

| Idea | Gap in Avizo | NUScope | Status |
|---|---|---|---|
| **Uncertainty on every measurement** | Measurements are single numbers | Sampling error (integral range for fractions, block bootstrap for S/V and tortuosity) plus segmentation-boundary sensitivity (±1 acquired voxel), reported as 95 % intervals | done |
| **Representative-volume check (RVE)** | No answer to "is my FIB volume big enough?" | Integral range from the phase covariance (Kanit et al. 2003), validated against independent realisations; reports the cube size needed for a target precision and how many RVEs the volume holds | done |
| **Data-health inspector** | Artifacts are found by eye | Per-slice SNR, sharpness, brightness drift, curtaining index, duplicate/missing slices, drift residuals, and an automatic calibration check that flags features stretched along the milling axis (the 50 → 20 nm error on sample 335 would have been caught) | done |
| **Segmentation uncertainty map** | Hard labels only | Per-voxel class probability and entropy from the classifier; shown as an overlay; drives active learning | next |
| Ground-truth test suite | Closed validation | Every analysis has an analytical or forward-model test that runs in CI | done |

## 2. While the microscope is still milling

| Idea | Gap | NUScope | Status |
|---|---|---|---|
| **Live acquisition mode** | Analysis starts after acquisition ends | `nuscope-tomo watch`: ingests slices as Auto Slice & View writes them, pre-processes and segments on the fly, updates Studio live, and reports when metrics reach RVE so milling can stop early | done |
| Acquisition advisor | None | From early slices: pixel size and slice thickness from feature sizes and an empirical coarsening test ("would coarser sampling have changed the answer?"), plus the depth / field of view needed for a target precision | done |
| **Context-driven sparse EDS** (after Kalinin, Ziatdinov et al.) | EDS maps every pixel | Deep kernel learning maps image context → composition and measures where it is least certain. With SE + BSE context, 10 % of the EDS time comes within ~1 % phase accuracy of a full map; SE-only context is blind to chemistry SE cannot see | done |
| **Pan-sharpened chemistry** (after Borodinov, Kalinin, Ovchinnikova et al. 2019) | — | Coupled-NMF fusion of a coarse EDS map with physically linked high-resolution channels (BSE via Castaing's rule) | done; next: full energy spectra (low-rank), ToF-SIMS |
| **Self-driving slice-and-view** (Thermo Fisher AutoScript) | Fixed Auto Slice & View recipes | Agent adapts focus, drift, dwell, slice thickness and sparse EDS, stops at the requested precision; AutoScript backend (dry-run by default, arming phrase, safety limits), digital twin and shadow mode for review before beam time | done (twin); hardware untested |

## 3. AI that is open and accountable

| Idea | Gap | NUScope | Status |
|---|---|---|---|
| **Self-supervised denoising (Noise2Void)** | Pre-trained or vendor-trained models | Trains on the dataset itself, no clean target needed, Apple MPS / CUDA | done |
| **Sparse-label U-Net** | Deep learning needs dense ground truth or a vendor model | Trains from the same scribbles as the random forest: the forest's confident predictions become weighted pseudo-labels, the 2.5D network adds context along Z; model saved with the project | done |
| Active learning | User guesses where to annotate | Studio highlights the slices and regions where the model is least certain | next |
| Natural-language recipes | Menu-driven | "Segment pores and compute tortuosity along Z" → validated YAML recipe from the operation registry (Claude API), reviewed before it runs | next |
| Model cards | Opaque models | Each trained model saved with its training data summary, features, metrics and provenance | next |

## 4. Beyond one dataset

| Idea | Gap | NUScope | Status |
|---|---|---|---|
| Cohort comparison | One project at a time | Compare samples with sampling-aware statistics, Holm correction, minimum detectable difference and the volume needed to detect it | done |
| Statistical descriptors + synthetic twins | Only measured volume | Two-point correlation, chord lengths; generate larger statistically equivalent volumes for simulation | next |
| FAIR export | Proprietary project files | One-click RO-Crate package: data, recipe, versions, report, ready for a DOI | next |
| Remote / shared Studio | Desktop licence per seat | Browser UI next to the data; multi-user annotation and comments at 3D positions | partly done |
