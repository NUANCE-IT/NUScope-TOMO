# NuScope Tomo

A desktop application for reconstructing and visualizing 3D volumes from
FIB-SEM (focused ion beam / scanning electron microscope) slice-and-view
image series — an open alternative to the reconstruction/visualization
workflow in software like Avizo, scoped to the FIB-SEM pipeline: import a
slice stack, align it, threshold-segment it, render it in 3D, and export a
surface mesh.

## Why it's built this way

FIB-SEM stacks can run past what fits in RAM, so the volume is never fully
loaded except at two deliberate points:

- **Storage**: every volume (raw / aligned / segmentation mask) lives on
  disk as a [zarr](https://zarr.dev) array and is only ever touched through
  [dask](https://dask.org), so import, slice-to-slice alignment and
  thresholding all run chunk-by-chunk with bounded memory, regardless of
  stack size.
- **Rendering / meshing**: GPU volume rendering and marching-cubes meshing
  need an in-memory array. `VolumeData.preview()` downsamples the whole
  volume to a memory-bounded size for these; `VolumeData.roi()` pulls a
  full-resolution crop when you need to inspect a region exactly. This is
  the same tradeoff Avizo/napari make for large data — it's made explicit
  here rather than hidden.

## Install

Requires Python 3.11+ (developed against 3.12; VTK/PyVista wheels lag
behind the newest CPython releases, so avoid 3.13+ until they catch up).

```bash
conda create -n nuscope-tomo python=3.12
conda activate nuscope-tomo
pip install -r requirements.txt
```

## Run

```bash
python -m nuscope_tomo.main
```

or, after `pip install -e .`:

```bash
nuscope-tomo
```

## Workflow

1. **File > Import Slice Stack…** — point it at a folder of 2D slice
   images (TIFF/PNG/BMP/JPEG, natural-sorted by trailing number in the
   filename) and a project folder to store the volume in, and set the
   voxel size (Z = slice thickness, Y/X = pixel size, in nm). This streams
   the images into `project/raw.zarr` one slice at a time.
2. **Process > Align Slices…** — corrects slice-to-slice drift with
   subpixel phase cross-correlation (registers each slice to its
   predecessor and accumulates the shift, since FIB-SEM drift is
   sequential, not fixed to one reference frame), then writes an aligned,
   auto-cropped copy to `project/aligned.zarr`.
3. **Process > Threshold Segmentation…** — pick a low/high intensity band
   (or click "Compute Otsu Threshold" for a data-driven default) and apply
   it out-of-core to produce `project/mask.zarr`.
4. **Process > Generate Mesh…** — downsamples the mask to an in-memory
   preview, removes small components, runs marching cubes, and prompts to
   export an STL/OBJ/PLY.
5. **File > Save Project** — persists whichever volumes exist (raw /
   aligned / mask) plus a `project.json` manifest; **File > Open
   Project…** reloads them.

The **Layers** dock controls what's shown in the 3D view (volume
rendering, orthogonal slice planes, per-mesh visibility) and which volume
is active; the 2D panel on the right lets you scrub through any single
Z/Y/X plane directly from disk.

## Project structure

```
nuscope_tomo/
  data/volume.py          VolumeData: dask/zarr-backed out-of-core volume
  io/stack_io.py           image sequence -> zarr import
  io/project_io.py         project save/load (project.json manifest)
  processing/alignment.py  phase cross-correlation slice registration
  processing/segmentation.py  Otsu / threshold / connected components
  processing/meshing.py    marching cubes + STL/OBJ/PLY export
  ui/                       PySide6 app: main window, PyVista 3D view,
                            2D slice browser, dialogs
tests/smoke_test.py         headless pipeline check on synthetic data
```

## Known limitations (first pass)

- Volume rendering and meshing operate on a downsampled preview for large
  volumes, not full resolution — bump the "3D preview resolution cap" in
  the Layers dock if your machine has the RAM/VRAM for more, or use
  `VolumeData.roi()` (currently only exposed programmatically) to inspect
  a full-resolution region.
- Connected-component labeling and mesh generation are in-memory
  operations; there's no chunked/distributed labeling across zarr chunk
  boundaries yet.
- No undo/redo, no manual (paint-brush) segmentation — thresholding only.
