"""Optional measurement inputs (angles + arch height).

Precision note
--------------
Arch height drives the deterministic classification via the brief's cm
bands, so it must accept 2-decimal precision (e.g. 4.69, not rounded to
4.7). All fields now use 2 decimals and a 0.01 step. The special
"unset" sentinel is the field's minimum value, shown as " — ".
"""

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

# key, label, unit, min, max, step, decimals, default(=min => "unset")
MEASUREMENTS = [
    ("calcaneal_inclination_deg", "Calcaneal inclination", "°", -90.0, 90.0, 0.01, 2, -90.0),
    ("heel_angle_deg",            "Heel angle",            "°", -45.0, 45.0, 0.01, 2, -45.0),
    ("arch_height_cm",            "Arch height",           "cm", 0.0,  15.0, 0.01, 2,  0.0),
    ("kite_angle_deg",            "Kite angle",            "°",  0.0,  90.0, 0.01, 2,  0.0),
    ("first_metatarsal_talus_deg","1st metatarsal–talus",  "°", -45.0, 45.0, 0.01, 2, -45.0),
]


class MeasurementPanel(QGroupBox):
    """Form for the 5 clinical measurements.

    When arch height is provided, classification is deterministic and
    authoritative (the brief's bands). Other fields are recorded for the
    report and used by the model's estimator when arch height is absent.
    """

    measurements_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__("Measurements", parent)

        self._spins: dict[str, QDoubleSpinBox] = {}
        self._unset_value: dict[str, float] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 24, 14, 14)
        outer.setSpacing(8)

        hint = QLabel(
            "Enter <b>arch height</b> for an authoritative, exact "
            "classification. Leave it blank to let the model estimate from "
            "images (assistive only). Other values improve the report."
        )
        hint.setStyleSheet("color: #9BA4B5; font-size: 11px;")
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addLayout(form)

        for key, label, unit, lo, hi, step, decimals, unset in MEASUREMENTS:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setSuffix(f" {unit}")
            spin.setValue(unset)
            spin.setSpecialValueText(" — ")     # shown when value == minimum
            spin.setMinimumWidth(150)
            spin.setKeyboardTracking(False)      # emit only on commit, not each keystroke
            spin.valueChanged.connect(self._emit)
            self._spins[key] = spin
            self._unset_value[key] = unset

            row_label = QLabel(label)
            if key == "arch_height_cm":
                row_label.setText(label + "  *")
                row_label.setStyleSheet("font-weight: 600;")
            form.addRow(row_label, spin)

        clear = QPushButton("Clear measurements")
        clear.clicked.connect(self.clear)
        outer.addWidget(clear)

    # ---------------------------------------------------------------- API
    def values(self) -> dict[str, float]:
        """Return only the fields the user actually set (not at sentinel)."""
        out: dict[str, float] = {}
        for key, spin in self._spins.items():
            if spin.value() != self._unset_value[key]:
                out[key] = float(spin.value())
        return out

    def clear(self) -> None:
        for key, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(self._unset_value[key])
            spin.blockSignals(False)
        self._emit()

    def set_values(self, values: dict[str, float]) -> None:
        for key, val in values.items():
            if key in self._spins and val is not None:
                self._spins[key].setValue(float(val))

    def _emit(self) -> None:
        self.measurements_changed.emit(self.values())
