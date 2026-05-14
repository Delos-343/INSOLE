"""
Classification tab — the PRIMARY tab.

Layout mirrors the "EXPECTED OUTPUT (in exe file)" diagram from the brief:

    +-------------------+-------------------+-------------------+
    |   INSERT IMAGE    |   INSERT IMAGE    |   INSERT IMAGE    |
    |                   |                   |                   |
    |  Lateral view     |   Top view        |   Back view       |
    +-------------------+-------------------+-------------------+
    |  [ Measurements ]                                         |
    |  [ Patient code ] [ ANALYZE FOOT ]                        |
    +-----------------------------------------------------------+

With the Results panel on the right side (split view).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_CONFIG
from app.ui.widgets.image_dropzone import ImageDropZone
from app.ui.widgets.measurement_panel import MeasurementPanel
from app.ui.widgets.results_panel import ResultsPanel
from app.ui.workers.inference_worker import InferenceWorker


class ClassificationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._measurements: dict = {}
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)

        # ---------- LEFT: inputs ----------
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(14)

        # Header
        header = QVBoxLayout()
        h_sub = QLabel("Classify")
        h_sub.setObjectName("subtitleLabel")
        h_title = QLabel("Foot image analysis")
        h_title.setObjectName("titleLabel")
        h_hint = QLabel(
            "Upload up to three foot views and (optionally) clinical "
            "measurements. The AI will classify the foot type and recommend "
            "an insole configuration."
        )
        h_hint.setStyleSheet("color: #9BA4B5; font-size: 12px;")
        h_hint.setWordWrap(True)
        header.addWidget(h_sub)
        header.addWidget(h_title)
        header.addWidget(h_hint)
        left_l.addLayout(header)

        # Image row: three drop zones
        img_row = QHBoxLayout()
        img_row.setSpacing(12)
        self.zone_lateral = ImageDropZone("Lateral view", "Side view of foot")
        self.zone_top     = ImageDropZone("Top view (AP)", "Top / dorsal view")
        self.zone_back    = ImageDropZone("Back view", "Posterior / heel view")
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            img_row.addWidget(z, 1)
            z.image_changed.connect(self._update_cta_state)
        left_l.addLayout(img_row, 1)

        # Measurements
        self.meas_panel = MeasurementPanel()
        self.meas_panel.measurements_changed.connect(self._on_measurements_changed)
        left_l.addWidget(self.meas_panel)

        # Patient code + CTA row
        cta_row = QHBoxLayout()
        cta_row.setSpacing(12)
        patient_label = QLabel("Patient code")
        patient_label.setObjectName("sectionLabel")
        self.patient_code = QLineEdit()
        self.patient_code.setPlaceholderText("e.g. P1097 (optional)")
        self.patient_code.setMaximumWidth(180)
        cta_row.addWidget(patient_label)
        cta_row.addWidget(self.patient_code)
        cta_row.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        cta_row.addWidget(self.clear_btn)

        self.cta = QPushButton("Analyze Foot")
        self.cta.setObjectName("primaryButton")
        self.cta.setEnabled(False)
        self.cta.clicked.connect(self._on_classify)
        cta_row.addWidget(self.cta)

        left_l.addLayout(cta_row)
        splitter.addWidget(left)

        # ---------- RIGHT: results ----------
        self.results = ResultsPanel()
        splitter.addWidget(self.results)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([800, 600])
        outer.addWidget(splitter, 1)

    # -------------------------------------------------------------- events
    def _on_measurements_changed(self, vals: dict) -> None:
        self._measurements = vals
        self._update_cta_state()

    def _update_cta_state(self, *_args) -> None:
        any_image = any(
            z.image_path for z in (self.zone_lateral, self.zone_top, self.zone_back)
        )
        self.cta.setEnabled(any_image)

    def _on_clear(self) -> None:
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            z.clear()
        self.meas_panel.clear()
        self.patient_code.clear()
        self.results.clear()

    def _on_classify(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self.cta.setEnabled(False)
        self.cta.setText("Analyzing…")

        self._worker = InferenceWorker(
            api_base_url=APP_CONFIG.api_base_url,
            lateral_path=self.zone_lateral.image_path,
            top_path=self.zone_top.image_path,
            back_path=self.zone_back.image_path,
            measurements=self._measurements,
            patient_code=(self.patient_code.text().strip() or None),
            use_local_fallback=APP_CONFIG.use_local_inference_fallback,
            parent=self,
        )
        self._worker.finished_ok.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._reset_cta)
        self._worker.start()

    def _on_result(self, result: dict) -> None:
        self.results.set_result(result)

    def _on_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Analysis failed", msg)

    def _reset_cta(self) -> None:
        self._update_cta_state()
        self.cta.setText("Analyze Foot")

    # ------------------------------------------------------------- helpers
    def set_images(self, lateral: Path | None, top: Path | None, back: Path | None) -> None:
        """Programmatic setter — used by demo button / drag from disk."""
        if lateral:
            self.zone_lateral.set_image(lateral)
        if top:
            self.zone_top.set_image(top)
        if back:
            self.zone_back.set_image(back)
