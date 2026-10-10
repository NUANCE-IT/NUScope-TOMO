from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)


class MeshDialog(QDialog):
    """Options for turning a segmentation mask into a surface mesh.

    Marching cubes needs the mask in memory, so large volumes are
    downsampled to `max_voxels` first (a preview, not full resolution) —
    that tradeoff is made explicit here rather than hidden.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Mesh")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.max_voxels_spin = QSpinBox()
        self.max_voxels_spin.setRange(32, 1000)
        self.max_voxels_spin.setValue(200)
        self.max_voxels_spin.setSuffix("³ voxels (preview cap, per axis)")
        form.addRow("Mesh resolution cap:", self.max_voxels_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 1_000_000)
        self.min_size_spin.setValue(64)
        form.addRow("Remove components smaller than (voxels):", self.min_size_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, int]:
        cap = self.max_voxels_spin.value()
        return cap**3, self.min_size_spin.value()
