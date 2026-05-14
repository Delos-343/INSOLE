"""Top-level QMainWindow."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from app.config import APP_CONFIG
from app.ui.tabs.classification_tab import ClassificationTab
from app.ui.tabs.training_tab import TrainingTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_CONFIG.app_name} — v{APP_CONFIG.app_version}")
        self.resize(APP_CONFIG.initial_width, APP_CONFIG.initial_height)

        self._build_menu()
        self._build_tabs()
        self._build_statusbar()

    # ------------------------------------------------------------------ ui
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_tabs(self) -> None:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.cls_tab = ClassificationTab()
        self.train_tab = TrainingTab()
        tabs.addTab(self.cls_tab, "Classification")
        tabs.addTab(self.train_tab, "Training")
        self.setCentralWidget(tabs)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.addPermanentWidget(QLabel(f"API: {APP_CONFIG.api_base_url}"))

    # --------------------------------------------------------------- about
    def _show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About",
            f"<h3>{APP_CONFIG.app_name}</h3>"
            f"<p>Version {APP_CONFIG.app_version}</p>"
            "<p>AI-powered foot classification system for insole "
            "recommendation, built on PyTorch + PySide6.</p>"
            "<p>Architecture: multi-view CNN + cross-modal fusion + "
            "generative VAE branch.</p>",
        )
