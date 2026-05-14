"""
/api/data — read-only inspection of the data folder.

GET  /api/data/summary  -> counts per class, per view, per patient
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from backend.model.data.dataset import IMAGE_EXTS, FootClassificationDataset

router = APIRouter()


@router.get("/summary")
async def summary() -> dict:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    out: dict[str, object] = {"data_dir": str(data_dir.resolve()), "exists": data_dir.exists()}

    if not data_dir.exists():
        return out

    # Top-level folder counts.
    folder_image_counts: dict[str, int] = {}
    for sub in ("Heel", "Flat", "Normal"):
        root = data_dir / sub
        if not root.exists():
            folder_image_counts[sub] = 0
            continue
        folder_image_counts[sub] = sum(
            1 for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
    out["folder_image_counts"] = folder_image_counts

    # Sheet files.
    sheet_dir = data_dir / "Sheet"
    out["sheet_files"] = (
        [p.name for p in sheet_dir.glob("*.xlsx")] if sheet_dir.exists() else []
    )

    # Try a dataset scan — gives per-class counts + total patients.
    try:
        ds = FootClassificationDataset(data_dir=data_dir, transform=None)
        out["total_samples"] = len(ds)
        out["class_counts"] = ds.class_counts()
        out["unique_patients"] = len({s.patient_id for s in ds.samples})
    except Exception as exc:
        out["scan_error"] = str(exc)

    return out
