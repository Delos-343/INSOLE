"""Qt stylesheet generator — produces the global QSS string."""

from __future__ import annotations

from app.ui.theme.colors import PALETTE as P


def build_stylesheet() -> str:
    """Compose the application-wide QSS."""
    return f"""
    /* =========================================================================
       Insole Foot Classification — Dark theme stylesheet
       ========================================================================= */

    QWidget {{
        background-color: {P.bg_primary};
        color: {P.text_primary};
        font-family: "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
        font-size: 13px;
    }}

    QMainWindow, QDialog {{
        background-color: {P.bg_primary};
    }}

    /* -------------------- Cards / Group boxes -------------------- */
    QGroupBox {{
        background-color: {P.bg_secondary};
        border: 1px solid {P.border};
        border-radius: 10px;
        margin-top: 18px;
        padding: 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
        color: {P.text_secondary};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    /* -------------------- Tabs -------------------- */
    QTabWidget::pane {{
        border: none;
        background: transparent;
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: transparent;
        color: {P.text_muted};
        padding: 10px 20px;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        font-size: 13px;
        letter-spacing: 0.3px;
    }}
    QTabBar::tab:hover {{
        color: {P.text_secondary};
    }}
    QTabBar::tab:selected {{
        color: {P.text_primary};
        border-bottom: 2px solid {P.accent};
    }}

    /* -------------------- Buttons -------------------- */
    QPushButton {{
        background-color: {P.bg_tertiary};
        color: {P.text_primary};
        border: 1px solid {P.border};
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
        min-height: 18px;
    }}
    QPushButton:hover {{
        background-color: {P.border};
        border-color: {P.accent_muted};
    }}
    QPushButton:pressed {{
        background-color: {P.bg_secondary};
    }}
    QPushButton:disabled {{
        color: {P.text_muted};
        background-color: {P.bg_secondary};
    }}

    QPushButton#primaryButton {{
        background-color: {P.accent};
        color: white;
        border: none;
        font-weight: 600;
        padding: 12px 28px;
        font-size: 14px;
        letter-spacing: 0.3px;
    }}
    QPushButton#primaryButton:hover {{
        background-color: {P.accent_hover};
    }}
    QPushButton#primaryButton:disabled {{
        background-color: {P.accent_muted};
        color: rgba(255,255,255,0.5);
    }}

    QPushButton#dangerButton {{
        background-color: transparent;
        color: {P.danger};
        border: 1px solid {P.danger};
    }}
    QPushButton#dangerButton:hover {{
        background-color: rgba(229,72,77,0.10);
    }}

    /* -------------------- Inputs -------------------- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background-color: {P.bg_tertiary};
        border: 1px solid {P.border};
        border-radius: 6px;
        padding: 7px 10px;
        selection-background-color: {P.accent_muted};
        color: {P.text_primary};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{
        border: 1px solid {P.accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {P.bg_secondary};
        border: 1px solid {P.border};
        selection-background-color: {P.accent_muted};
        color: {P.text_primary};
    }}

    /* -------------------- Labels -------------------- */
    QLabel#titleLabel {{
        font-size: 24px;
        font-weight: 700;
        color: {P.text_primary};
        letter-spacing: -0.5px;
    }}
    QLabel#subtitleLabel {{
        font-size: 12px;
        color: {P.text_muted};
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    QLabel#metricLabel {{
        font-size: 11px;
        color: {P.text_muted};
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    QLabel#metricValue {{
        font-size: 28px;
        font-weight: 700;
        color: {P.text_primary};
    }}
    QLabel#sectionLabel {{
        font-size: 11px;
        color: {P.text_secondary};
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }}

    /* -------------------- ProgressBar -------------------- */
    QProgressBar {{
        background-color: {P.bg_tertiary};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
        color: {P.text_primary};
    }}
    QProgressBar::chunk {{
        background-color: {P.accent};
        border-radius: 4px;
    }}

    /* -------------------- ScrollBar -------------------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {P.border};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {P.accent_muted};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
    }}
    QScrollBar::handle:horizontal {{
        background: {P.border};
        border-radius: 5px;
        min-width: 30px;
    }}

    /* -------------------- StatusBar -------------------- */
    QStatusBar {{
        background-color: {P.bg_secondary};
        color: {P.text_muted};
        border-top: 1px solid {P.border};
    }}

    /* -------------------- Drag-and-drop zone -------------------- */
    QFrame#dropZone {{
        background-color: {P.bg_secondary};
        border: 2px dashed {P.border};
        border-radius: 12px;
    }}
    QFrame#dropZone[active="true"] {{
        border: 2px dashed {P.accent};
        background-color: {P.bg_tertiary};
    }}
    QFrame#dropZone[filled="true"] {{
        border: 1px solid {P.accent_muted};
        background-color: {P.bg_secondary};
    }}

    /* -------------------- Splitter -------------------- */
    QSplitter::handle {{
        background-color: {P.border};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}

    /* -------------------- Menu -------------------- */
    QMenuBar {{
        background-color: {P.bg_secondary};
        color: {P.text_secondary};
        border-bottom: 1px solid {P.border};
    }}
    QMenuBar::item:selected {{
        background-color: {P.bg_tertiary};
    }}
    QMenu {{
        background-color: {P.bg_secondary};
        border: 1px solid {P.border};
        color: {P.text_primary};
    }}
    QMenu::item:selected {{
        background-color: {P.accent_muted};
    }}
    """
