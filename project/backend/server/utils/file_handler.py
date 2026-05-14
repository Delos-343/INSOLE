"""Helpers for receiving and persisting image uploads."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))


def _check_ext(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_EXT)}",
        )
    return suffix


async def save_upload(file: UploadFile, view_tag: str = "img") -> Path:
    """Persist an UploadFile under UPLOAD_DIR. Return the path."""
    suffix = _check_ext(file.filename or "upload")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{view_tag}_{uuid.uuid4().hex[:12]}{suffix}"
    target = UPLOAD_DIR / fname

    size_bytes = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    with target.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum size of {MAX_UPLOAD_MB} MB.",
                )
            out.write(chunk)
    return target
