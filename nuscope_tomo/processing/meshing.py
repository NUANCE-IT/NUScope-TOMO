"""Isosurface mesh generation from a segmented volume.

Marching cubes needs the whole array in memory, so this operates on an
in-memory label/mask array (a `VolumeData.preview()` or `VolumeData.roi()`
result) rather than the out-of-core volume directly — the caller decides
the memory/resolution tradeoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def generate_mesh(
    mask: np.ndarray,
    voxel_size_nm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.5,
    step_size: int = 1,
) -> trimesh.Trimesh:
    """Run marching cubes on a binary/label mask and return a mesh in
    physical units (nanometers), respecting anisotropic voxel spacing."""
    if mask.dtype != bool:
        mask = mask > 0
    if not mask.any():
        raise ValueError("Mask is empty — nothing to mesh")

    verts, faces, normals, _values = marching_cubes(
        mask.astype(np.uint8),
        level=level,
        spacing=voxel_size_nm,
        step_size=step_size,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Export to STL/OBJ/PLY — format chosen from the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path
