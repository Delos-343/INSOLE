"""Read-only log console for the Training tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class LogConsole(QPlainTextEdit):
    """Auto-scrolling monospace console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        f = QFont("JetBrains Mono", 10)
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(f)
        self.setStyleSheet(
            "QPlainTextEdit { background-color: #0B0E14; color: #B6C2D9; "
            "border: 1px solid #222A38; border-radius: 8px; padding: 8px; }"
        )

    def append_line(self, text: str) -> None:
        self.appendPlainText(text.rstrip())
        self.moveCursor(QTextCursor.MoveOperation.End)
