"""Optional measurement inputs (angles + arch height)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

MEASUREMENTS = [
    ("calcaneal_inclination_deg", "Calcaneal inclination", "°", -90.0, 90.0, 0.1, 0.0),
    ("heel_angle_deg",            "Heel angle",            "°", -45.0, 45.0, 0.1, 0.0),
    ("arch_height_cm",            "Arch height",           "cm", 0.0,  15.0, 0.1, 0.0),
    ("kite_angle_deg",            "Kite angle",            "°",  0.0,  90.0, 0.1, 0.0),
    ("first_metatarsal_talus_deg","1st metatarsal–talus",  "°", -45.0, 45.0, 0.1, 0.0),
]


class MeasurementPanel(QGroupBox):
    """Form for the 5 clinical measurements. Optional — model can predict."""

    measurements_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__("Measurements (optional)", parent)

        self._spins: dict[str, QDoubleSpinBox] = {}
        self._enabled: dict[str, bool] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 24, 14, 14)
        outer.setSpacing(8)

        hint = QLabel("Provide any subset — the AI will estimate the rest from the images.")
        hint.setStyleSheet("color: #9BA4B5; font-size: 11px;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addLayout(form)

        for key, label, unit, lo, hi, step, default in MEASUREMENTS:
            row = QFrame()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(8)

            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(1)
            spin.setSuffix(f" {unit}")
            spin.setValue(default)
            spin.setSpecialValueText(" — ")  # show when at minimum
            spin.setMinimumWidth(140)
            spin.valueChanged.connect(self._emit)
            row_l.addWidget(spin, 1)
            self._spins[key] = spin
            self._enabled[key] = False

            form.addRow(QLabel(label), row)

        # Quick clear
        clear = QPushButton("Clear measurements")
        clear.clicked.connect(self.clear)
        outer.addWidget(clear)

    # ---------------------------------------------------------------- API
    def values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, spin in self._spins.items():
            if spin.value() != spin.minimum():
                out[key] = float(spin.value())
        return out

    def clear(self) -> None:
        for spin in self._spins.values():
            spin.blockSignals(True)
            spin.setValue(spin.minimum())
            spin.blockSignals(False)
        self._emit()

    def set_values(self, values: dict[str, float]) -> None:
        for key, val in values.items():
            if key in self._spins and val is not None:
                self._spins[key].setValue(float(val))

    def _emit(self) -> None:
        self.measurements_changed.emit(self.values())
