"""
ImageDropZone — a tile that lets the user drop or click-to-browse an image.

Looks like the "INSERT IMAGE" placeholders in the brief's EXPECTED OUTPUT
mockup, with a small caption underneath (Lateral / Top / Back).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QVBoxLayout,
)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDropZone(QFrame):
    """A drag-and-drop / click-to-browse image tile."""

    image_changed = Signal(str)  # absolute path

    def __init__(self, view_label: str, hint: str = "Drop image here or click", parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._view_label = view_label
        self._hint = hint
        self._image_path: str | None = None
        self.setMinimumSize(QSize(220, 220))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Preview takes most of the space.
        self.preview = QLabel("INSERT\nIMAGE")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setStyleSheet(
            "color: #5D6779;"
            "font-size: 13px;"
            "letter-spacing: 2px;"
            "font-weight: 600;"
        )
        layout.addWidget(self.preview, 1)

        # Caption (Lateral / Top / Back) - matches brief mockup.
        self.caption = QLabel(self._view_label)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet(
            "color: #E6EAF2; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;"
        )
        layout.addWidget(self.caption)

        # Small hint
        self.hint_label = QLabel(self._hint)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #5D6779; font-size: 10px;")
        layout.addWidget(self.hint_label)

    # ------------------------------------------------------------------ API
    @property
    def image_path(self) -> str | None:
        return self._image_path

    def clear(self) -> None:
        self._image_path = None
        self.preview.setText("INSERT\nIMAGE")
        self.preview.setPixmap(QPixmap())
        self.setProperty("filled", False)
        self._refresh_style()
        self.image_changed.emit("")

    def set_image(self, path: str | Path) -> None:
        path_str = str(path)
        pix = QPixmap(path_str)
        if pix.isNull():
            return
        self._image_path = path_str
        self._update_preview(pix)
        self.setProperty("filled", True)
        self._refresh_style()
        self.image_changed.emit(path_str)

    def _update_preview(self, pix: QPixmap) -> None:
        self.preview.setPixmap(
            pix.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _refresh_style(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)

    # ------------------------------------------------------ DnD + click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {self._view_label} view image",
                "",
                "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
            )
            if file_path:
                self.set_image(file_path)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._mime_has_image(event.mimeData()):
            self.setProperty("active", True)
            self._refresh_style()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.setProperty("active", False)
        self._refresh_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("active", False)
        self._refresh_style()
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if Path(local).suffix.lower() in ALLOWED_EXT:
                self.set_image(local)
                event.acceptProposedAction()
                return
        event.ignore()

    def resizeEvent(self, event):
        if self._image_path:
            pix = QPixmap(self._image_path)
            if not pix.isNull():
                self._update_preview(pix)
        super().resizeEvent(event)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _mime_has_image(mime: QMimeData) -> bool:
        if not mime.hasUrls():
            return False
        return any(Path(u.toLocalFile()).suffix.lower() in ALLOWED_EXT for u in mime.urls())
