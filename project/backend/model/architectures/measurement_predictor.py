"""
Measurement-regression head — predicts the 5 clinical angles/heights from
the fused image embedding alone.

This is the "X-ray-free" path: if the user uploads only photos and no
measurements, the network estimates them itself so the GUI can still
display the metrics panel from the brief.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeasurementHead(nn.Module):
    """Regresses the 5 measurements from a fused embedding."""

    MEASUREMENT_NAMES: tuple[str, ...] = (
        "calcaneal_inclination_deg",
        "heel_angle_deg",
        "arch_height_cm",
        "kite_angle_deg",
        "first_metatarsal_talus_deg",
    )

    # Approximate plausible ranges, used to re-scale the sigmoid output.
    # These intentionally span well beyond clinical extremes.
    MEASUREMENT_RANGES: tuple[tuple[float, float], ...] = (
        (10.0, 30.0),   # calcaneal inclination (deg)
        (-15.0, 20.0),  # heel angle (deg)
        (1.0, 8.0),     # arch height (cm)
        (10.0, 45.0),   # kite angle (deg)
        (-10.0, 25.0),  # 1st metatarsal-talus angle (deg)
    )

    def __init__(self, fusion_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        n = len(self.MEASUREMENT_NAMES)
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n),
        )
        self.register_buffer(
            "lows",
            torch.tensor([lo for lo, _ in self.MEASUREMENT_RANGES], dtype=torch.float32),
        )
        self.register_buffer(
            "highs",
            torch.tensor([hi for _, hi in self.MEASUREMENT_RANGES], dtype=torch.float32),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns predicted measurements in their natural units."""
        raw = self.net(fused)
        gated = torch.sigmoid(raw)
        return self.lows + (self.highs - self.lows) * gated
