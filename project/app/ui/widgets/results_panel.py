"""
ResultsPanel — right-hand results pane.

Now leads with a PROVENANCE BANNER that makes the trust level
unmistakable:

  • MEASURED (green)   — deterministic classification from clinician
                         measurements. Authoritative. Confidence 100%.
  • ESTIMATED (amber)  — arch height was estimated from images because no
                         measurement was supplied. Assistive only;
                         diagnostic testing showed image-only estimates
                         are unreliable for this dataset.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme.colors import PALETTE as P


def _severity_color(band: str) -> str:
    return {
        "normal": P.severity_normal,
        "moderate": P.severity_moderate,
        "severe": P.severity_severe,
    }.get(band, P.text_muted)


class ResultsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsPanel")
        self.setStyleSheet(
            f"#resultsPanel {{ background-color: {P.bg_secondary}; "
            f"border: 1px solid {P.border}; border-radius: 12px; }}"
        )
        self._build()
        self.clear()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("Results")
        title.setObjectName("subtitleLabel")
        outer.addWidget(title)

        # ---- Provenance banner (the trust signal) ----
        self.banner = QLabel("—")
        self.banner.setWordWrap(True)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setMinimumHeight(46)
        outer.addWidget(self.banner)

        self.headline = QLabel("—")
        self.headline.setObjectName("titleLabel")
        outer.addWidget(self.headline)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(10)
        self.severity_chip = QLabel("—")
        self.severity_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.severity_chip.setMinimumWidth(90)
        self.severity_chip.setStyleSheet(self._chip_qss(P.text_muted))
        chip_row.addWidget(self.severity_chip)
        self.confidence_label = QLabel("Confidence —")
        self.confidence_label.setStyleSheet(
            f"color: {P.text_secondary}; font-size: 13px;"
        )
        chip_row.addWidget(self.confidence_label)
        chip_row.addStretch()
        outer.addLayout(chip_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(0, 4, 0, 0)
        body_l.setSpacing(16)

        # Probabilities
        probs_box = QGroupBox("Class probabilities")
        probs_l = QVBoxLayout(probs_box)
        probs_l.setContentsMargins(14, 24, 14, 14)
        probs_l.setSpacing(6)
        self.prob_rows: dict[str, tuple[QLabel, QProgressBar, QLabel]] = {}
        for cls in ("Severe Flat Arch", "Flat Arch", "Normal Foot",
                    "High Arch", "Severe High Arch"):
            row = QHBoxLayout()
            row.setSpacing(10)
            name = QLabel(cls)
            name.setMinimumWidth(140)
            name.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            val = QLabel("0.0%")
            val.setMinimumWidth(50)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet(f"color: {P.text_primary};")
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(val)
            probs_l.addLayout(row)
            self.prob_rows[cls] = (name, bar, val)
        body_l.addWidget(probs_box)

        # Measurements used (with per-row provenance tag)
        self.meas_box = QGroupBox("Measurements used for classification")
        meas_l = QVBoxLayout(self.meas_box)
        meas_l.setContentsMargins(14, 24, 14, 14)
        meas_l.setSpacing(4)
        self.meas_labels: dict[str, QLabel] = {}
        self._meas_keys = [
            ("calcaneal_inclination_deg", "Calcaneal inclination", "°"),
            ("heel_angle_deg", "Heel angle", "°"),
            ("arch_height_cm", "Arch height", "cm"),
            ("kite_angle_deg", "Kite angle", "°"),
            ("first_metatarsal_talus_deg", "1st metatarsal–talus", "°"),
        ]
        for k, lbl, unit in self._meas_keys:
            row = QHBoxLayout()
            row.setSpacing(8)
            l = QLabel(lbl)
            l.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
            v = QLabel(f"— {unit}")
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            v.setStyleSheet(f"color: {P.text_primary}; font-size: 13px;")
            self.meas_labels[k] = v
            row.addWidget(l)
            row.addStretch()
            row.addWidget(v)
            meas_l.addLayout(row)
        body_l.addWidget(self.meas_box)

        # Insole config
        insole_box = QGroupBox("Recommended insole configuration")
        in_l = QVBoxLayout(insole_box)
        in_l.setContentsMargins(14, 24, 14, 14)
        in_l.setSpacing(6)
        self.insole_rows: dict[str, QProgressBar] = {}
        for k, lbl in [
            ("arch_support_height", "Arch support height"),
            ("heel_cup_depth", "Heel cup depth"),
            ("medial_post_strength", "Medial post strength"),
            ("lateral_wedge_strength", "Lateral wedge strength"),
            ("forefoot_cushioning", "Forefoot cushioning"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(10)
            l = QLabel(lbl)
            l.setMinimumWidth(160)
            l.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            row.addWidget(l)
            row.addWidget(bar, 1)
            in_l.addLayout(row)
            self.insole_rows[k] = bar
        body_l.addWidget(insole_box)

        self.notes_label = QLabel("")
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet(
            f"color: {P.text_muted}; font-size: 11px; font-style: italic;"
        )
        body_l.addWidget(self.notes_label)
        body_l.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    def _chip_qss(self, color: str) -> str:
        return (
            f"background-color: {color}22; color: {color}; font-weight: 700; "
            f"font-size: 10px; letter-spacing: 2px; text-transform: uppercase; "
            f"border: 1px solid {color}66; border-radius: 999px; padding: 4px 12px;"
        )

    def _banner_qss(self, color: str) -> str:
        return (
            f"background-color: {color}1F; color: {color}; "
            f"border: 1px solid {color}80; border-radius: 8px; "
            f"padding: 10px 14px; font-size: 12px; font-weight: 600;"
        )

    def clear(self) -> None:
        self.banner.setText("Awaiting input")
        self.banner.setStyleSheet(self._banner_qss(P.text_muted))
        self.headline.setText("—")
        self.severity_chip.setText("—")
        self.severity_chip.setStyleSheet(self._chip_qss(P.text_muted))
        self.confidence_label.setText("Confidence —")
        for _, bar, val in self.prob_rows.values():
            bar.setValue(0)
            val.setText("0.0%")
        for v in self.meas_labels.values():
            v.setText("—")
        for bar in self.insole_rows.values():
            bar.setValue(0)
        self.notes_label.setText("")

    def set_result(self, result: dict) -> None:
        source = result.get("classification_source", "measured")
        cls = result.get("predicted_class", "—")
        conf = float(result.get("confidence", 0.0))
        severity = result.get("severity_band", "unknown")

        if source == "measured":
            self.banner.setText(
                "✓  MEASURED — authoritative result.\n"
                "Classified deterministically from the provided clinical "
                "measurements using the project's arch-height bands."
            )
            self.banner.setStyleSheet(self._banner_qss(P.success))
            self.meas_box.setTitle("Measurements used (clinician-provided)")
        else:
            self.banner.setText(
                "⚠  ESTIMATED — assistive only, NOT a clinical measurement.\n"
                "No arch-height value was supplied, so it was estimated from "
                "the images. Image-only estimates are unreliable for this "
                "dataset — confirm with a real measurement before use."
            )
            self.banner.setStyleSheet(self._banner_qss(P.warning))
            self.meas_box.setTitle("Measurements used (model-estimated)")

        self.headline.setText(cls)
        self.confidence_label.setText(f"Confidence {conf * 100:0.1f}%")
        color = _severity_color(severity)
        self.severity_chip.setText(severity)
        self.severity_chip.setStyleSheet(self._chip_qss(color))

        probs = result.get("class_probabilities") or {}
        for name, (_, bar, val) in self.prob_rows.items():
            p = float(probs.get(name, 0.0))
            bar.setValue(int(round(p * 100)))
            val.setText(f"{p * 100:0.1f}%")
            chunk = color if name == cls else P.accent_muted
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {P.bg_tertiary}; border: none; "
                f"border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background-color: {chunk}; "
                f"border-radius: 3px; }}"
            )

        units = {
            "calcaneal_inclination_deg": "°",
            "heel_angle_deg": "°",
            "arch_height_cm": "cm",
            "kite_angle_deg": "°",
            "first_metatarsal_talus_deg": "°",
        }
        used = result.get("measurements_predicted") or {}
        provided = result.get("measurements_provided") or {}
        for key, lbl in self.meas_labels.items():
            v = used.get(key)
            unit = units[key]
            if v is None:
                lbl.setText(f"— {unit}")
                continue
            tag = " (measured)" if provided.get(key) is not None else " (est.)"
            lbl.setText(f"{v:.2f} {unit}{tag}")

        insole = result.get("insole_configuration") or {}
        for key, bar in self.insole_rows.items():
            bar.setValue(int(round(float(insole.get(key, 0.0)) * 100)))

        notes = result.get("notes") or []
        self.notes_label.setText("• " + "\n• ".join(notes) if notes else "")
