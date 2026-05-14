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
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata or {}},
        tmp,
    )
    tmp.replace(path)
    logger.debug("Saved checkpoint -> {}", path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (state_dict, metadata)."""
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("metadata", {})
    return payload, {}
