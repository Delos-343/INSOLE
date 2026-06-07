"""
Application entry point — `python -m app.main` or the bundled exe.
"""

import app.bootstrap_streams  # noqa: F401  -- MUST be first, before any heavy import

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from app.config import APP_CONFIG
from app.ui.main_window import MainWindow
from app.ui.theme.stylesheet import build_stylesheet


def _force_dark_palette(app: QApplication) -> None:
    """Set a base dark palette so native dialogs match the QSS."""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
    pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
    pal.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.black)
    pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(pal)


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_CONFIG.app_name)
    app.setApplicationVersion(APP_CONFIG.app_version)
    app.setOrganizationName("Insole Foot Classification")
    app.setStyle("Fusion")

    if APP_CONFIG.use_dark_mode:
        _force_dark_palette(app)
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
