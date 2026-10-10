"""2D orthogonal slice browser: reads one plane at a time straight from the
(possibly out-of-core) volume, so scrubbing through an 800-slice stack never
touches more than one image.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...data.volume import VolumeData
from . import colormaps

_AXES = {"Z (slice)": 0, "Y": 1, "X": 2}


class SliceView2D(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.volume: VolumeData | None = None
        self._buffer: bytes | None = None
        self.cmap_name = colormaps.DEFAULT_INTENSITY_COLORMAP

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(list(_AXES.keys()))
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        controls.addWidget(self.axis_combo)

        self.index_slider = QSlider(Qt.Horizontal)
        self.index_slider.valueChanged.connect(self._on_index_changed)
        controls.addWidget(self.index_slider, stretch=1)

        self.index_spin = QSpinBox()
        self.index_spin.valueChanged.connect(self.index_slider.setValue)
        self.index_slider.valueChanged.connect(self.index_spin.setValue)
        controls.addWidget(self.index_spin)
        root.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel("No volume loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        root.addWidget(self.scroll, stretch=1)

    def set_volume(self, volume: VolumeData) -> None:
        self.volume = volume
        self._on_axis_changed()

    def set_colormap(self, display_name: str) -> None:
        self.cmap_name = display_name
        self._refresh()

    def _on_axis_changed(self) -> None:
        if self.volume is None:
            return
        axis = _AXES[self.axis_combo.currentText()]
        size = self.volume.shape[axis]
        self.index_slider.blockSignals(True)
        self.index_spin.blockSignals(True)
        self.index_slider.setRange(0, size - 1)
        self.index_spin.setRange(0, size - 1)
        self.index_slider.setValue(size // 2)
        self.index_spin.setValue(size // 2)
        self.index_slider.blockSignals(False)
        self.index_spin.blockSignals(False)
        self._refresh()

    def _on_index_changed(self, _value: int) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.volume is None:
            return
        axis = _AXES[self.axis_combo.currentText()]
        index = self.index_slider.value()
        plane = self.volume.get_slice(axis, index)
        pixmap = self._to_pixmap(plane)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def _to_pixmap(self, plane: np.ndarray) -> QPixmap:
        plane = np.asarray(plane)
        lo, hi = float(plane.min()), float(plane.max())
        if hi > lo:
            scaled = ((plane.astype(np.float32) - lo) / (hi - lo) * 255.0)
        else:
            scaled = np.zeros_like(plane, dtype=np.float32)
        buf = np.ascontiguousarray(scaled.astype(np.uint8))
        h, w = buf.shape

        if colormaps.mpl_name(self.cmap_name) == "gray":
            self._buffer = buf.tobytes()  # keep alive: QImage doesn't copy by default
            image = QImage(self._buffer, w, h, w, QImage.Format_Grayscale8)
        else:
            lut = colormaps.lut_for(self.cmap_name)
            rgb = np.ascontiguousarray(lut[buf])  # (h, w, 3) uint8
            self._buffer = rgb.tobytes()
            image = QImage(self._buffer, w, h, w * 3, QImage.Format_RGB888)

        return QPixmap.fromImage(image)
