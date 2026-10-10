from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..data.volume import VolumeData, VoxelSize
from ..io.project_io import load_project, save_project
from ..io.stack_io import import_image_stack
from ..processing.alignment import run_alignment
from ..processing.meshing import export_mesh, generate_mesh
from ..processing.segmentation import apply_threshold, label_components
from .widgets import colormaps
from .widgets.import_dialog import ImportDialog
from .widgets.mesh_dialog import MeshDialog
from .widgets.segmentation_dialog import SegmentationDialog
from .widgets.slice_view import SliceView2D
from .widgets.volume_view import VolumeView3D
from .workers import Worker


def _build_mesh_from_mask(volume: VolumeData, max_voxels: int, min_size: int):
    """Runs in a background thread: downsample to an in-memory preview,
    clean up small components, marching-cube it. Spacing is corrected for
    however much the preview was downsampled relative to the source."""
    preview = volume.preview(max_voxels=max_voxels)
    labeled = label_components(preview, min_size=min_size)
    mask = labeled > 0

    vz, vy, vx = volume.voxel_size.as_tuple()
    fz = volume.shape[0] / preview.shape[0]
    fy = volume.shape[1] / preview.shape[1]
    fx = volume.shape[2] / preview.shape[2]
    spacing = (vz * fz, vy * fy, vx * fx)

    return generate_mesh(mask, voxel_size_nm=spacing)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NuScope Tomo")
        self.resize(1500, 950)

        self.project_dir: Path | None = None
        self.volumes: dict[str, VolumeData] = {}
        self.meshes: dict[str, object] = {}
        self.active_key: str | None = None
        self._worker: Worker | None = None
        self._progress: QProgressDialog | None = None

        self.volume_view = VolumeView3D()
        self.slice_view = SliceView2D()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.volume_view)
        splitter.addWidget(self.slice_view)
        splitter.setSizes([950, 500])
        self.setCentralWidget(splitter)

        self._build_menu()
        self._build_layers_dock()

        self.status = self.statusBar()
        self.status.showMessage("No volume loaded — File > Import Slice Stack…")

    # -- menu / docks ---------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Import Slice Stack…", self._import_stack)
        file_menu.addAction("Open Project…", self._open_project)
        file_menu.addAction("Save Project", self._save_project)
        file_menu.addAction("Save Project As…", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        process_menu = self.menuBar().addMenu("&Process")
        process_menu.addAction("Align Slices…", self._align_slices)
        process_menu.addAction("Threshold Segmentation…", self._segment)
        process_menu.addAction("Generate Mesh…", self._generate_mesh)

    def _build_layers_dock(self) -> None:
        dock = QDockWidget("Layers", self)
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(QLabel("Active volume:"))
        self.volume_combo = QComboBox()
        self.volume_combo.currentTextChanged.connect(self._on_active_volume_changed)
        layout.addWidget(self.volume_combo)

        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(colormaps.COLORMAPS.keys()))
        self.colormap_combo.setCurrentText(colormaps.DEFAULT_INTENSITY_COLORMAP)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        self.show_volume_chk = QCheckBox("Volume rendering")
        self.show_volume_chk.setChecked(True)
        self.show_volume_chk.toggled.connect(lambda _: self._refresh_3d())
        layout.addWidget(self.show_volume_chk)

        self.show_slices_chk = QCheckBox("Orthogonal slice planes")
        self.show_slices_chk.toggled.connect(lambda _: self._refresh_3d())
        layout.addWidget(self.show_slices_chk)

        layout.addWidget(QLabel("3D preview resolution cap:"))
        self.preview_cap_spin = QSpinBox()
        self.preview_cap_spin.setRange(32, 400)
        self.preview_cap_spin.setValue(160)
        self.preview_cap_spin.setSuffix("³ voxels")
        self.preview_cap_spin.valueChanged.connect(lambda _: self._refresh_3d())
        layout.addWidget(self.preview_cap_spin)

        layout.addWidget(QLabel("Meshes:"))
        self.mesh_list = QListWidget()
        self.mesh_list.itemChanged.connect(self._on_mesh_item_changed)
        layout.addWidget(self.mesh_list, stretch=1)

        dock.setWidget(content)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # -- background work helper -----------------------------------------------
    def _start_worker(self, fn, *args, on_success, title, **kwargs) -> None:
        self._worker = Worker(fn, *args, **kwargs)
        self._progress = QProgressDialog(title, None, 0, 0, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setCancelButton(None)

        def on_progress(done: int, total: int) -> None:
            if self._progress.maximum() != total:
                self._progress.setMaximum(total)
            self._progress.setValue(done)

        def on_finished(result) -> None:
            self._progress.close()
            on_success(result)

        def on_failed(message: str) -> None:
            self._progress.close()
            QMessageBox.critical(self, "Error", message)

        self._worker.progress.connect(on_progress)
        self._worker.finished_ok.connect(on_finished)
        self._worker.failed.connect(on_failed)
        self._progress.show()
        self._worker.start()

    # -- import / project ---------------------------------------------------
    def _import_stack(self) -> None:
        dialog = ImportDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        source = Path(values["source_folder"])
        project = values["project_folder"]

        if not source.is_dir():
            QMessageBox.warning(self, "Import", "Choose a valid source folder.")
            return
        if not project:
            QMessageBox.warning(self, "Import", "Choose a project folder to store the volume in.")
            return

        self.project_dir = Path(project)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        voxel_size = VoxelSize(*values["voxel_size"])
        output_path = self.project_dir / "raw.zarr"

        self._start_worker(
            import_image_stack,
            source,
            output_path,
            voxel_size=voxel_size,
            name=values["name"],
            on_success=self._on_import_done,
            title="Importing slice stack…",
        )

    def _on_import_done(self, volume: VolumeData) -> None:
        self.volumes = {"raw": volume}
        self.meshes.clear()
        self.mesh_list.clear()
        self.active_key = "raw"
        self._refresh_volume_combo()
        self._show_active_volume()
        self.status.showMessage(
            f"Imported '{volume.name}': shape={volume.shape}, "
            f"voxel={volume.voxel_size.as_tuple()} nm"
        )

    def _save_project(self) -> None:
        if self.project_dir is None or not self.volumes:
            QMessageBox.warning(self, "Save Project", "Nothing to save yet.")
            return
        self.volumes = save_project(self.project_dir, self.volumes, active=self.active_key)
        self.status.showMessage(f"Project saved to {self.project_dir}")

    def _save_project_as(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Save project as…")
        if not path:
            return
        self.project_dir = Path(path)
        self._save_project()

    def _open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open project")
        if not path:
            return
        try:
            volumes, active, _extra = load_project(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Open Project", str(exc))
            return

        self.project_dir = Path(path)
        self.volumes = volumes
        self.meshes.clear()
        self.mesh_list.clear()
        self.active_key = active or next(iter(volumes), None)
        self._refresh_volume_combo()
        self._show_active_volume()
        self.status.showMessage(f"Loaded project {path}")

    # -- processing -----------------------------------------------------------
    def _align_slices(self) -> None:
        if "raw" not in self.volumes:
            QMessageBox.warning(self, "Align", "Import a slice stack first.")
            return
        output_path = self.project_dir / "aligned.zarr"
        self._start_worker(
            run_alignment,
            self.volumes["raw"],
            output_path,
            on_success=self._on_align_done,
            title="Aligning slices (phase cross-correlation)…",
        )

    def _on_align_done(self, volume: VolumeData) -> None:
        self.volumes["aligned"] = volume
        self.active_key = "aligned"
        self._refresh_volume_combo()
        self._show_active_volume()
        self.status.showMessage(f"Alignment complete: shape={volume.shape}")

    def _segment(self) -> None:
        if not self.active_key:
            QMessageBox.warning(self, "Segment", "Load or import a volume first.")
            return
        volume = self.volumes[self.active_key]
        dialog = SegmentationDialog(volume, self)
        if dialog.exec() != QDialog.Accepted:
            return
        low, high = dialog.values()
        output_path = self.project_dir / "mask.zarr"
        self._start_worker(
            apply_threshold,
            volume,
            low,
            high,
            output_path,
            on_success=self._on_segment_done,
            title="Applying threshold…",
        )

    def _on_segment_done(self, volume: VolumeData) -> None:
        self.volumes["mask"] = volume
        self.active_key = "mask"
        self._refresh_volume_combo()
        self._show_active_volume()
        self.status.showMessage(f"Segmentation complete: shape={volume.shape}")

    def _generate_mesh(self) -> None:
        volume = self.volumes.get("mask") or (
            self.volumes.get(self.active_key) if self.active_key else None
        )
        if volume is None:
            QMessageBox.warning(self, "Mesh", "Run threshold segmentation first.")
            return
        dialog = MeshDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        max_voxels, min_size = dialog.values()
        self._start_worker(
            _build_mesh_from_mask,
            volume,
            max_voxels,
            min_size,
            on_success=self._on_mesh_done,
            title="Generating mesh (marching cubes)…",
        )

    def _on_mesh_done(self, mesh) -> None:
        name = f"mesh_{len(self.meshes) + 1}"
        self.meshes[name] = mesh
        self.volume_view.add_mesh(mesh, name)

        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.mesh_list.addItem(item)

        self.status.showMessage(
            f"Mesh '{name}': {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        default_path = str(self.project_dir / f"{name}.stl") if self.project_dir else f"{name}.stl"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Mesh", default_path, "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)"
        )
        if path:
            export_mesh(mesh, path)
            self.status.showMessage(f"Mesh '{name}' exported to {path}")

    def _on_mesh_item_changed(self, item: QListWidgetItem) -> None:
        self.volume_view.set_mesh_visible(item.text(), item.checkState() == Qt.Checked)

    # -- view state -----------------------------------------------------------
    def _refresh_volume_combo(self) -> None:
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        self.volume_combo.addItems(list(self.volumes.keys()))
        if self.active_key:
            self.volume_combo.setCurrentText(self.active_key)
        self.volume_combo.blockSignals(False)

    def _on_active_volume_changed(self, key: str) -> None:
        if not key or key not in self.volumes:
            return
        self.active_key = key
        # Segmentation masks are label data (adjacent integers are different
        # objects), not a gradient, so default to a qualitative colormap —
        # but only as a suggestion; the user's own pick below still wins.
        suggested = (
            colormaps.DEFAULT_LABEL_COLORMAP
            if key == "mask"
            else colormaps.DEFAULT_INTENSITY_COLORMAP
        )
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText(suggested)
        self.colormap_combo.blockSignals(False)
        self._show_active_volume()

    def _on_colormap_changed(self, _name: str) -> None:
        self.slice_view.set_colormap(self.colormap_combo.currentText())
        self._refresh_3d()

    def _show_active_volume(self) -> None:
        if not self.active_key:
            return
        self.slice_view.set_colormap(self.colormap_combo.currentText())
        self.slice_view.set_volume(self.volumes[self.active_key])
        self._refresh_3d()

    def _refresh_3d(self) -> None:
        if not self.active_key:
            return
        volume = self.volumes[self.active_key]
        cap = self.preview_cap_spin.value() ** 3
        preview = volume.preview(max_voxels=cap)

        # preview() downsamples each axis by a (possibly non-uniform, due to
        # rounding) integer factor — derive the effective spacing from the
        # actual shape ratio rather than reusing the full-resolution voxel
        # size, or the preview renders too small relative to its true extent.
        vz, vy, vx = volume.voxel_size.as_tuple()
        fz = volume.shape[0] / preview.shape[0]
        fy = volume.shape[1] / preview.shape[1]
        fx = volume.shape[2] / preview.shape[2]
        voxel = (vz * fz, vy * fy, vx * fx)
        cmap = colormaps.mpl_name(self.colormap_combo.currentText())

        if self.show_volume_chk.isChecked():
            self.volume_view.show_volume(preview, voxel, cmap=cmap)
        else:
            self.volume_view.clear_volume()

        if self.show_slices_chk.isChecked():
            self.volume_view.show_orthogonal_slices(preview, voxel, cmap=cmap)
        else:
            self.volume_view.clear_slices()

    def closeEvent(self, event) -> None:
        self.volume_view.plotter.close()
        super().closeEvent(event)
