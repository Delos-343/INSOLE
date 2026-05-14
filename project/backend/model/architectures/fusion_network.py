"""
Cross-modal fusion network.

Combines:
  - Three view embeddings (lateral, top, back), each of shape (B, D_view)
  - Tabular measurements (calcaneal incl., heel angle, arch height, kite,
    1st metatarsal-talus), shape (B, M)

Strategy:
  1. Project each view embedding to a common dimension.
  2. Apply self-attention across the three view tokens so they can
     contextualise each other (multi-view cross-talk).
  3. Encode the tabular measurements with an MLP into the same token space.
  4. Concatenate all four tokens (3 views + 1 measurement) and apply a
     final cross-token attention block before pooling.
  5. Return a single fused embedding of shape (B, fusion_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small transformer encoder block
# ---------------------------------------------------------------------------
class _TokenTransformerBlock(nn.Module):
    """Pre-norm transformer block on a small set of tokens."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_tokens, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Fusion module
# ---------------------------------------------------------------------------
class MultiModalFusion(nn.Module):
    """Cross-modal fusion of 3 image views and a measurement vector."""

    def __init__(
        self,
        view_dims: tuple[int, int, int],
        num_measurements: int,
        fusion_dim: int = 512,
        num_attention_blocks: int = 2,
    ) -> None:
        super().__init__()

        # Project each view's embedding to the shared fusion dimension.
        self.lateral_proj = nn.Linear(view_dims[0], fusion_dim)
        self.top_proj = nn.Linear(view_dims[1], fusion_dim)
        self.back_proj = nn.Linear(view_dims[2], fusion_dim)

        # Encode tabular measurements -> token of size fusion_dim.
        self.measurement_encoder = nn.Sequential(
            nn.Linear(num_measurements, fusion_dim // 2),
            nn.GELU(),
            nn.LayerNorm(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Learned positional/role embeddings: tells the transformer
        # which token represents which modality.
        self.role_embeddings = nn.Parameter(torch.zeros(4, fusion_dim))
        nn.init.trunc_normal_(self.role_embeddings, std=0.02)

        # Stack of transformer blocks operating on the 4 tokens.
        self.blocks = nn.ModuleList(
            [_TokenTransformerBlock(fusion_dim) for _ in range(num_attention_blocks)]
        )

        self.norm = nn.LayerNorm(fusion_dim)
        self.fusion_dim = fusion_dim

    def forward(
        self,
        lateral_feat: torch.Tensor,
        top_feat: torch.Tensor,
        back_feat: torch.Tensor,
        measurements: torch.Tensor,
        measurement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            lateral_feat:    ``(B, D_lat)``
            top_feat:        ``(B, D_top)``
            back_feat:       ``(B, D_back)``
            measurements:    ``(B, M)`` — angles/heights, can be zero-valued.
            measurement_mask:``(B, 1)`` 1.0 if measurements provided, else 0.0.
                             When 0.0, the measurement token is replaced with
                             a learned "missing" embedding (handled outside).

        Returns:
            ``(B, fusion_dim)`` pooled fused representation.
        """
        b = lateral_feat.size(0)

        lat = self.lateral_proj(lateral_feat)        # (B, D)
        top = self.top_proj(top_feat)                # (B, D)
        bak = self.back_proj(back_feat)              # (B, D)
        meas = self.measurement_encoder(measurements)  # (B, D)

        if measurement_mask is not None:
            # Zero out measurement contribution where mask == 0 — the
            # role embedding still tells the model "this slot is meas".
            meas = meas * measurement_mask.view(b, 1)

        # Stack -> (B, 4, D)
        tokens = torch.stack([lat, top, bak, meas], dim=1)

        # Add role embeddings -> tokens know which modality they are.
        tokens = tokens + self.role_embeddings.unsqueeze(0)

        # Cross-modal attention.
        for blk in self.blocks:
            tokens = blk(tokens)

        # Mean-pool across modalities for the final representation.
        pooled = self.norm(tokens).mean(dim=1)
        return pooled
