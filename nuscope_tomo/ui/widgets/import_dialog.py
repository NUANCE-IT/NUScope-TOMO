from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class ImportDialog(QDialog):
    """Collects: source image folder, destination project folder, voxel
    size (nm) and a display name, before importing a FIB-SEM slice stack."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Slice Stack")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.source_edit = QLineEdit()
        form.addRow("Source folder (SE images):", self._with_browse(self.source_edit, self._browse_source))

        self.project_edit = QLineEdit()
        form.addRow("New project folder:", self._with_browse(self.project_edit, self._browse_project))

        self.name_edit = QLineEdit("sample")
        form.addRow("Sample name:", self.name_edit)

        self.voxel_z = self._nm_spin(50.0)
        self.voxel_y = self._nm_spin(5.0)
        self.voxel_x = self._nm_spin(5.0)
        form.addRow("Slice thickness Z (nm):", self.voxel_z)
        form.addRow("Pixel size Y (nm):", self.voxel_y)
        form.addRow("Pixel size X (nm):", self.voxel_x)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _nm_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.001, 100000.0)
        spin.setDecimals(3)
        spin.setValue(default)
        return spin

    def _with_browse(self, line_edit: QLineEdit, slot) -> "QWidget":
        from PySide6.QtWidgets import QWidget

        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return container

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder of slice images")
        if path:
            self.source_edit.setText(path)
            if not self.name_edit.text() or self.name_edit.text() == "sample":
                self.name_edit.setText(Path(path).name)

    def _browse_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select (or create) project folder")
        if path:
            self.project_edit.setText(path)

    def values(self) -> dict:
        return {
            "source_folder": self.source_edit.text(),
            "project_folder": self.project_edit.text(),
            "name": self.name_edit.text() or "sample",
            "voxel_size": (self.voxel_z.value(), self.voxel_y.value(), self.voxel_x.value()),
        }
