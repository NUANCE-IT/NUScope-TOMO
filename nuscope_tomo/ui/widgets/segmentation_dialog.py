from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...data.volume import VolumeData
from ...processing.segmentation import otsu_threshold


class SegmentationDialog(QDialog):
    """Pick a low/high intensity band to threshold. 'Compute Otsu' derives
    a default cut from an out-of-core (subsampled) histogram."""

    def __init__(self, volume: VolumeData, parent=None):
        super().__init__(parent)
        self.volume = volume
        self.setWindowTitle("Threshold Segmentation")
        self.setMinimumWidth(360)

        info = np.iinfo(volume.dtype) if np.issubdtype(volume.dtype, np.integer) else None
        vmax = float(info.max) if info is not None else 1.0

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.low_spin = QDoubleSpinBox()
        self.low_spin.setRange(0, vmax)
        self.low_spin.setValue(vmax * 0.5)
        form.addRow("Low threshold:", self.low_spin)

        self.high_spin = QDoubleSpinBox()
        self.high_spin.setRange(0, vmax)
        self.high_spin.setValue(vmax)
        form.addRow("High threshold:", self.high_spin)

        layout.addLayout(form)

        otsu_btn = QPushButton("Compute Otsu Threshold")
        otsu_btn.clicked.connect(self._compute_otsu)
        layout.addWidget(otsu_btn)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _compute_otsu(self) -> None:
        self.status_label.setText("Computing…")
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        value = otsu_threshold(self.volume)
        self.low_spin.setValue(value)
        self.status_label.setText(f"Otsu threshold: {value:.2f}")

    def values(self) -> tuple[float, float]:
        return self.low_spin.value(), self.high_spin.value()
