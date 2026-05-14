"""
Generative branch — a conditional VAE that operates in the fused-embedding
space rather than in pixel space.

Purpose
-------
1. **Synthetic embedding augmentation** during training: sample new latent
   vectors conditioned on a target class and decode them back into the
   fusion space, then push them through the classifier head. This is a
   form of feature-space mixup that regularises the classifier with very
   little extra cost.

2. **Insole "pattern" generation** at inference time: given a class and an
   optional measurement vector, we can sample a representative latent
   point and decode it to a fusion vector. That vector is interpreted by
   downstream code (the GUI/API) as a low-dim insole-shape signature,
   e.g. arch-support height, heel-cup depth, medial post angle — these
   are read out by a separate light decoder ``InsoleConfigHead``.

The VAE is intentionally compact; it's a regulariser, not a renderer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFusionVAE(nn.Module):
    """A conditional VAE that lives in the fusion-embedding space."""

    def __init__(
        self,
        fusion_dim: int = 512,
        latent_dim: int = 128,
        num_classes: int = 5,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ---- Encoder: fused_embedding + class_one_hot -> q(z|x,y) ----
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim + num_classes, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---- Decoder: z + class_one_hot -> fused_embedding ----
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, fusion_dim),
        )

    # ---------------------------------------------------------------- enc/dec
    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, y_onehot], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, y_onehot], dim=-1))

    # ---------------------------------------------------------------- forward
    def forward(
        self, fused: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fused:  ``(B, fusion_dim)``  fused embedding from the main net.
            labels: ``(B,)`` int64 class indices.

        Returns:
            (reconstruction, mu, logvar)
        """
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        mu, logvar = self.encode(fused, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        num_per_label: int = 1,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Sample synthetic fused embeddings for the given class indices."""
        labels = labels.repeat_interleave(num_per_label).to(device)
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        z = torch.randn(labels.size(0), self.latent_dim, device=device) * temperature
        return self.decode(z, y_onehot)


# ---------------------------------------------------------------------------
# Loss term
# ---------------------------------------------------------------------------
def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> torch.Tensor:
    """MSE reconstruction + beta * KL divergence (beta-VAE style)."""
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


# ---------------------------------------------------------------------------
# Tiny readout head: fused-embedding -> insole config vector
# ---------------------------------------------------------------------------
class InsoleConfigHead(nn.Module):
    """Reads the fused embedding and emits a 5-dim insole configuration.

    Output channels (all normalised to [0,1] via sigmoid):
        0 - Arch support height
        1 - Heel cup depth
        2 - Medial post strength
        3 - Lateral wedge strength
        4 - Forefoot cushioning density
    """

    OUTPUT_NAMES: tuple[str, ...] = (
        "arch_support_height",
        "heel_cup_depth",
        "medial_post_strength",
        "lateral_wedge_strength",
        "forefoot_cushioning",
    )

    def __init__(self, fusion_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.OUTPUT_NAMES)),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)
