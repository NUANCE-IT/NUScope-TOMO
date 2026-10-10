"""End-to-end pipeline check with synthetic data, no Qt involved: generates
a drifting slice stack, imports it, aligns it, thresholds it, meshes it,
and exports an STL. Run with:

    python tests/smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuscope_tomo.data.volume import VoxelSize
from nuscope_tomo.io.project_io import load_project, save_project
from nuscope_tomo.io.stack_io import import_image_stack
from nuscope_tomo.processing.alignment import run_alignment
from nuscope_tomo.processing.meshing import export_mesh, generate_mesh
from nuscope_tomo.processing.segmentation import apply_threshold, label_components, otsu_threshold


def make_synthetic_stack(folder: Path, n=20, h=96, w=96) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:n, 0:h, 0:w]
    center = np.array([n / 2, h / 2, w / 2])
    for i in range(n):
        # a sphere that drifts a couple pixels per slice, plus noise
        drift_y, drift_x = i * 0.7, i * -0.4
        r2 = (
            (zz[i] - center[0]) ** 2
            + (yy[i] - center[1] - drift_y) ** 2
            + (xx[i] - center[2] - drift_x) ** 2
        )
        sphere = (r2 < (h / 4) ** 2).astype(np.uint8) * 200
        noise = rng.integers(0, 30, size=(h, w), dtype=np.uint8)
        frame = np.clip(sphere + noise, 0, 255).astype(np.uint8)
        tifffile.imwrite(folder / f"slice{i:03d}.tiff", frame)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nuscope_smoke_"))
    try:
        source = workdir / "source_images"
        project = workdir / "project"
        make_synthetic_stack(source)

        print("Importing...")
        raw = import_image_stack(
            source, project / "raw.zarr", voxel_size=VoxelSize(z=50, y=5, x=5), name="synthetic"
        )
        assert raw.shape == (20, 96, 96), raw.shape
        assert raw.array.dtype == np.uint8

        print("Reading a slice and a histogram out-of-core...")
        plane = raw.get_slice(0, 5)
        assert plane.shape == (96, 96)
        hist, edges = raw.histogram(bins=64)
        assert hist.sum() > 0

        print("Aligning...")
        aligned = run_alignment(raw, project / "aligned.zarr")
        assert aligned.shape[0] == raw.shape[0]
        assert aligned.shape[1] <= raw.shape[1] and aligned.shape[2] <= raw.shape[2]

        print("Thresholding...")
        threshold = otsu_threshold(aligned)
        mask = apply_threshold(aligned, threshold, 255, project / "mask.zarr")
        assert mask.array.dtype == np.uint8
        assert set(np.unique(mask.array[:2].compute())).issubset({0, 1})

        print("Meshing...")
        preview = mask.preview(max_voxels=64**3)
        labeled = label_components(preview, min_size=10)
        assert labeled.max() >= 1, "expected at least one connected component"
        mesh = generate_mesh(labeled > 0, voxel_size_nm=aligned.voxel_size.as_tuple())
        assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
        stl_path = export_mesh(mesh, project / "meshes" / "sphere.stl")
        assert stl_path.exists() and stl_path.stat().st_size > 0

        print("Saving and reloading project...")
        save_project(project, {"raw": raw, "aligned": aligned, "mask": mask}, active="aligned")
        volumes, active, _extra = load_project(project)
        assert active == "aligned"
        assert set(volumes.keys()) == {"raw", "aligned", "mask"}
        assert volumes["aligned"].shape == aligned.shape

        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
