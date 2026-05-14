"""
Per-view CNN feature extractor.

Each foot view (lateral, top, back) goes through its own backbone so the
model can learn view-specific features (e.g. arch curvature vs heel angle).
The backbone is sourced from `timm` so any vision model is swappable via
the config.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # graceful fallback for environments without timm
    _HAS_TIMM = False


class ViewEncoder(nn.Module):
    """A single-view image encoder returning a pooled feature vector.

    Args:
        backbone_name: timm model name (e.g. ``efficientnet_b0``).
        pretrained:    Initialise from ImageNet weights.
        out_dim:       Optionally project the pooled features to this size.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()

        if not _HAS_TIMM:
            raise ImportError(
                "`timm` is required for ViewEncoder. Install it with `pip install timm`."
            )

        # `num_classes=0` strips the classifier head -> we get pooled features.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # `feature_dim` auto-detected from the backbone.
        self.feature_dim: int = self.backbone.num_features

        # Optional projection head.
        self.projection: nn.Module
        if out_dim is not None and out_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )
            self.feature_dim = out_dim
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 3, H, W)`` image batch.

        Returns:
            ``(B, feature_dim)`` pooled embedding.
        """
        feats = self.backbone(x)
        feats = self.projection(feats)
        return feats
