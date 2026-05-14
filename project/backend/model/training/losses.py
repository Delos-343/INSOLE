"""Loss functions for the multi-task classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.model.architectures.generative_vae import vae_loss


class FocalLoss(nn.Module):
    """Focal loss — helpful when classes are imbalanced or hard to discriminate."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Combines classification, measurement regression, and VAE losses."""

    def __init__(
        self,
        num_classes: int,
        cls_weight: float = 1.0,
        meas_weight: float = 0.5,
        vae_weight: float = 0.1,
        class_weights: torch.Tensor | None = None,
        use_focal: bool = False,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.meas_weight = meas_weight
        self.vae_weight = vae_weight

        if use_focal:
            self.cls_loss_fn: nn.Module = FocalLoss(alpha=class_weights)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        self.meas_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        outputs: dict,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # ---------- classification ----------
        cls_loss = self.cls_loss_fn(outputs["logits"], batch["label"])
        total = self.cls_weight * cls_loss
        logs = {"cls_loss": float(cls_loss.item())}

        # ---------- measurement regression (only on rows that have meas) ----------
        if "measurements_hat" in outputs and self.meas_weight > 0:
            target = batch["measurements"]
            mask = batch["measurement_mask"]
            elementwise = self.meas_loss_fn(outputs["measurements_hat"], target)  # (B, 5)
            meas_loss = (elementwise.mean(dim=-1) * mask.squeeze(-1)).sum() / (
                mask.sum().clamp_min(1.0)
            )
            total = total + self.meas_weight * meas_loss
            logs["meas_loss"] = float(meas_loss.item())

        # ---------- VAE reconstruction + KL ----------
        if "vae" in outputs and self.vae_weight > 0:
            v = outputs["vae"]
            kl = vae_loss(v["recon"], outputs["fused"].detach(), v["mu"], v["logvar"])
            total = total + self.vae_weight * kl
            logs["vae_loss"] = float(kl.item())

        logs["total_loss"] = float(total.item())
        return total, logs
