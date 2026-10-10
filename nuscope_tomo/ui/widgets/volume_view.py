"""3D viewport: volume rendering, orthogonal slice planes and mesh overlays,
built on PyVista/VTK embedded in Qt via pyvistaqt.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


def _to_vtk_grid(array_zyx: np.ndarray, voxel_size_zyx: tuple[float, float, float]) -> pv.ImageData:
    """VolumeData stores arrays as (Z, Y, X) — the natural order for a slice
    stack. `pv.wrap()` on a 3D numpy array maps array axis 0 -> VTK's
    geometric X, axis 1 -> Y, axis 2 -> Z (verified empirically, not
    documented). Passed straight through, that put our slice-stack axis (Z)
    on the viewer's X and the image's X axis on the viewer's Z — a 90°
    permutation that, combined with the large Z/XY anisotropy typical of
    FIB-SEM, made reconstructions look visibly skewed. Transpose to (X, Y, Z)
    first so array axis 2 (our X) lines up with VTK X, etc.
    """
    array_xyz = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)))
    vz, vy, vx = voxel_size_zyx
    grid = pv.wrap(array_xyz)
    grid.spacing = (vx, vy, vz)
    return grid


def trimesh_to_pyvista(mesh) -> pv.PolyData:
    """`mesh.vertices` come from marching_cubes on a (Z, Y, X)-ordered array,
    so vertex columns are (Z, Y, X) physical coordinates. PyVista/VTK treat
    point columns as geometric (X, Y, Z), so permute columns to match.
    skimage's marching_cubes already emits winding that is "inward" under a
    naive read of those columns as XYZ (verified empirically: signed volume
    is negative before any permutation) — permuting two axes reverses
    orientation once, which exactly corrects it to outward-facing. Do not
    also reverse face winding here, or it cancels this out and normals end
    up pointing inward again.
    """
    verts_zyx = np.asarray(mesh.vertices)
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    faces = np.asarray(mesh.faces)
    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(verts_xyz, pv_faces)


class VolumeView3D(QWidget):
    """Owns one QtInteractor and tracks named actors so callers can
    show/hide/replace a volume, an orthogonal-slice set, and any number of
    named surface meshes independently."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_axes()

        self._volume_actor = None
        self._slice_actor = None
        self._mesh_actors: dict[str, object] = {}

    # -- volume rendering ---------------------------------------------------
    def show_volume(
        self,
        array: np.ndarray,
        voxel_size: tuple[float, float, float],
        cmap: str = "bone",
        opacity: str = "sigmoid",
    ) -> None:
        self.clear_volume()
        grid = _to_vtk_grid(array, voxel_size)
        self._volume_actor = self.plotter.add_volume(
            grid, cmap=cmap, opacity=opacity, name="volume", shade=True
        )
        self.plotter.reset_camera()

    def clear_volume(self) -> None:
        if self._volume_actor is not None:
            self.plotter.remove_actor(self._volume_actor)
            self._volume_actor = None

    # -- orthogonal slice planes ---------------------------------------------
    def show_orthogonal_slices(
        self, array: np.ndarray, voxel_size: tuple[float, float, float], cmap: str = "gray"
    ) -> None:
        self.clear_slices()
        grid = _to_vtk_grid(array, voxel_size)
        slices = grid.slice_orthogonal()
        self._slice_actor = self.plotter.add_mesh(slices, cmap=cmap, name="ortho_slices")
        self.plotter.reset_camera()

    def clear_slices(self) -> None:
        if self._slice_actor is not None:
            self.plotter.remove_actor(self._slice_actor)
            self._slice_actor = None

    # -- surface meshes -------------------------------------------------------
    def add_mesh(self, mesh, name: str, color: str = "tan", opacity: float = 1.0) -> None:
        self.remove_mesh(name)
        pv_mesh = trimesh_to_pyvista(mesh)
        actor = self.plotter.add_mesh(
            pv_mesh, color=color, opacity=opacity, name=name, smooth_shading=True
        )
        self._mesh_actors[name] = actor
        self.plotter.reset_camera()

    def remove_mesh(self, name: str) -> None:
        if name in self._mesh_actors:
            self.plotter.remove_actor(self._mesh_actors.pop(name))

    def set_mesh_visible(self, name: str, visible: bool) -> None:
        actor = self._mesh_actors.get(name)
        if actor is not None:
            actor.SetVisibility(visible)
            self.plotter.render()

    def clear_all(self) -> None:
        self.plotter.clear_actors()
        self.plotter.add_axes()
        self._volume_actor = None
        self._slice_actor = None
        self._mesh_actors.clear()
