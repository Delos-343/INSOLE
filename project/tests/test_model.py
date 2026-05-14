"""Smoke tests for the model architecture."""

from __future__ import annotations

import pytest
import torch

from backend.model.architectures.classifier import build_classifier
from backend.model.config import CLASS_NAMES, ModelConfig


@pytest.fixture(scope="module")
def model():
    cfg = ModelConfig(pretrained=False, lateral_backbone="resnet18",
                      top_backbone="resnet18", back_backbone="resnet18")
    return build_classifier(cfg)


def test_forward_shapes(model):
    B = 2
    out = model(
        lateral=torch.randn(B, 3, 224, 224),
        top=torch.randn(B, 3, 224, 224),
        back=torch.randn(B, 3, 224, 224),
        measurements=torch.zeros(B, 5),
        measurement_mask=torch.zeros(B, 1),
    )
    assert out["logits"].shape == (B, len(CLASS_NAMES))
    assert out["insole_config"].shape == (B, 5)
    assert out["measurements_hat"].shape == (B, 5)


def test_vae_branch(model):
    B = 2
    out = model(
        lateral=torch.randn(B, 3, 224, 224),
        top=torch.randn(B, 3, 224, 224),
        back=torch.randn(B, 3, 224, 224),
        measurements=torch.zeros(B, 5),
        measurement_mask=torch.zeros(B, 1),
        labels=torch.zeros(B, dtype=torch.long),
        return_vae=True,
    )
    assert "vae" in out
    assert out["vae"]["recon"].shape == (B, ModelConfig().fusion_dim)


def test_insole_config_bounded(model):
    """Insole sigmoid outputs must live in [0, 1]."""
    B = 4
    out = model(
        lateral=torch.randn(B, 3, 224, 224),
        top=torch.randn(B, 3, 224, 224),
        back=torch.randn(B, 3, 224, 224),
        measurements=torch.randn(B, 5),
        measurement_mask=torch.ones(B, 1),
    )
    cfg = out["insole_config"]
    assert (cfg >= 0).all() and (cfg <= 1).all()
