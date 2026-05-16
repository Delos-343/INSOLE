"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state + arbitrary metadata atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, tmp)
    tmp.replace(path)
    logger.debug("Saved checkpoint -> {}", path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (state_dict, metadata)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("metadata", {})
    return payload, {}


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the best available checkpoint.

    PRIORITY (not just mtime — that bug caused epoch_20.pt to be picked
    over the genuinely-best best.pt):
      1. best.pt                — the early-stopping best, ALWAYS preferred
      2. run_<uuid>.pt          — archived completed-run copies, newest first
      3. epoch_<n>.pt           — periodic snapshots, highest epoch first
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return None

    best = d / "best.pt"
    if best.is_file() and best.stat().st_size > 0:
        return best

    runs = sorted(
        (p for p in d.glob("run_*.pt") if p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if runs:
        return runs[0]

    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    epochs = sorted(
        (p for p in d.glob("epoch_*.pt") if p.stat().st_size > 0),
        key=_epoch_num,
        reverse=True,
    )
    if epochs:
        return epochs[0]

    any_pt = sorted(
        (p for p in d.glob("*.pt") if p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return any_pt[0] if any_pt else None
