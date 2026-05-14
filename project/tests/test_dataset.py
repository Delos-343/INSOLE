"""Tests for the dataset / data discovery logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.model.config import CLASS_NAMES
from backend.model.data.dataset import FootClassificationDataset


def _make_image(path: Path, size: int = 64, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


@pytest.fixture
def synthetic_data(tmp_path: Path) -> Path:
    """Build a tiny Drive-shaped tree with two patients."""
    root = tmp_path / "data"

    # Patient 1 — flat foot via folder convention
    p1 = root / "Flat" / "P1001"
    _make_image(p1 / "lat_view.jpg")
    _make_image(p1 / "top_view.jpg")
    # Heel image lives in the Heel/ tree
    _make_image(root / "Heel" / "H1-99" / "P1001" / "back_post.jpg")

    # Patient 2 — high arch, label inferred from sheet measurement
    p2 = root / "Normal" / "P1002"
    _make_image(p2 / "lat_side.jpg")

    sheet_dir = root / "Sheet"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "patient_id": ["P1001", "P1002"],
            "arch_height_cm": [3.0, 6.0],     # flat / high
            "heel_angle_deg": [8.0, -2.0],
        }
    ).to_excel(sheet_dir / "measurements.xlsx", index=False)
    return root


def test_dataset_discovers_patients(synthetic_data: Path) -> None:
    ds = FootClassificationDataset(data_dir=synthetic_data, transform=None)
    assert len(ds) == 2
    pids = {s.patient_id for s in ds.samples}
    assert pids == {"P1001", "P1002"}


def test_arch_height_drives_label(synthetic_data: Path) -> None:
    ds = FootClassificationDataset(data_dir=synthetic_data, transform=None)
    label_by_pid = {s.patient_id: s.label for s in ds.samples}
    assert label_by_pid["P1001"] == "Flat Arch"
    assert label_by_pid["P1002"] == "High Arch"


def test_getitem_returns_expected_keys(synthetic_data: Path) -> None:
    ds = FootClassificationDataset(data_dir=synthetic_data, transform=None)
    sample = ds[0]
    for key in ("lateral", "top", "back", "measurements", "measurement_mask",
                "view_mask", "label", "patient_id"):
        assert key in sample
    assert sample["lateral"].shape == (3, ds.image_size, ds.image_size)
    assert sample["measurements"].shape == (5,)


def test_class_counts(synthetic_data: Path) -> None:
    ds = FootClassificationDataset(data_dir=synthetic_data, transform=None)
    counts = ds.class_counts()
    assert sum(counts.values()) == 2
    assert all(n in CLASS_NAMES for n in counts)
