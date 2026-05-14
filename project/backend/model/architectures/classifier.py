"""
The top-level multi-view, multi-modal foot classifier.

Forward signature
-----------------
Inputs:
  - lateral:        (B, 3, H, W)
  - top:            (B, 3, H, W)
  - back:           (B, 3, H, W)
  - measurements:   (B, 5) — calcaneal incl., heel angle, arch height,
                              kite angle, 1st metatarsal-talus angle
  - measurement_mask: (B, 1) — 1.0 if user supplied measurements, else 0.0
  - labels:         (B,) optional, used only if `return_vae=True`

Outputs (dict):
  - logits:           (B, num_classes)
  - fused:            (B, fusion_dim)
  - measurements_hat: (B, 5) if `predict_measurements` is enabled
  - insole_config:    (B, 5) — sigmoid-bounded insole shape readout
  - vae:              (recon, mu, logvar) — only when return_vae=True
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from backend.model.architectures.fusion_network import MultiModalFusion
from backend.model.architectures.generative_vae import ConditionalFusionVAE, InsoleConfigHead
from backend.model.architectures.measurement_predictor import MeasurementHead
from backend.model.architectures.view_encoder import ViewEncoder
from backend.model.config import ModelConfig


class MultiViewFootClassifier(nn.Module):
    """Multi-view foot classifier with optional generative + regression heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ----- Per-view backbones -----
        self.lateral_encoder = ViewEncoder(cfg.lateral_backbone, pretrained=cfg.pretrained)
        self.top_encoder = ViewEncoder(cfg.top_backbone, pretrained=cfg.pretrained)
        self.back_encoder = ViewEncoder(cfg.back_backbone, pretrained=cfg.pretrained)

        # ----- Fusion -----
        self.fusion = MultiModalFusion(
            view_dims=(
                self.lateral_encoder.feature_dim,
                self.top_encoder.feature_dim,
                self.back_encoder.feature_dim,
            ),
            num_measurements=cfg.num_measurements,
            fusion_dim=cfg.fusion_dim,
        )

        # ----- Classification head -----
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout / 2),
            nn.Linear(cfg.fusion_dim // 2, cfg.num_classes),
        )

        # ----- Optional auxiliary heads -----
        self.measurement_head: MeasurementHead | None = (
            MeasurementHead(cfg.fusion_dim) if cfg.predict_measurements else None
        )
        self.vae: ConditionalFusionVAE | None = (
            ConditionalFusionVAE(
                fusion_dim=cfg.fusion_dim,
                latent_dim=cfg.vae_latent_dim,
                num_classes=cfg.num_classes,
            )
            if cfg.use_generative_branch
            else None
        )
        self.insole_head = InsoleConfigHead(cfg.fusion_dim)

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        lateral: torch.Tensor,
        top: torch.Tensor,
        back: torch.Tensor,
        measurements: torch.Tensor,
        measurement_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_vae: bool = False,
    ) -> dict[str, Any]:
        # 1) Encode each view independently.
        f_lat = self.lateral_encoder(lateral)
        f_top = self.top_encoder(top)
        f_bak = self.back_encoder(back)

        # 2) Cross-modal fusion.
        fused = self.fusion(f_lat, f_top, f_bak, measurements, measurement_mask)

        # 3) Heads.
        logits = self.classifier(fused)
        insole_config = self.insole_head(fused)

        out: dict[str, Any] = {
            "logits": logits,
            "fused": fused,
            "insole_config": insole_config,
        }

        if self.measurement_head is not None:
            out["measurements_hat"] = self.measurement_head(fused)

        if return_vae and self.vae is not None and labels is not None:
            recon, mu, logvar = self.vae(fused, labels)
            out["vae"] = {"recon": recon, "mu": mu, "logvar": logvar}

        return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_classifier(cfg: ModelConfig | None = None) -> MultiViewFootClassifier:
    """Factory that returns a freshly constructed classifier."""
    return MultiViewFootClassifier(cfg or ModelConfig())
