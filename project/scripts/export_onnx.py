"""
Export the trained model to ONNX for cross-platform deployment.

    python scripts/export_onnx.py --checkpoint backend/model/checkpoints/best.pt \\
                                  --output backend/model/checkpoints/foot_classifier.onnx
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from loguru import logger

from backend.model.architectures.classifier import build_classifier
from backend.model.config import ModelConfig
from backend.model.utils.checkpoint import load_checkpoint

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., help="Path to trained checkpoint .pt"),
    output: Path = typer.Option(..., help="Output .onnx path"),
    image_size: int = typer.Option(256),
    opset: int = typer.Option(17),
) -> None:
    state, meta = load_checkpoint(checkpoint, map_location="cpu")
    cfg = ModelConfig(**meta.get("model_cfg", {})) if meta else ModelConfig(pretrained=False)
    model = build_classifier(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Dummy inputs matching the forward() signature.
    B = 1
    dummy = {
        "lateral":          torch.randn(B, 3, image_size, image_size),
        "top":              torch.randn(B, 3, image_size, image_size),
        "back":             torch.randn(B, 3, image_size, image_size),
        "measurements":     torch.zeros(B, 5),
        "measurement_mask": torch.zeros(B, 1),
    }

    # ONNX export doesn't support **kwargs, so wrap.
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, lateral, top, back, measurements, measurement_mask):
            out = self.m(lateral, top, back, measurements, measurement_mask)
            return out["logits"], out["insole_config"], out.get("measurements_hat", out["logits"])

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrapper(model),
        tuple(dummy.values()),
        str(output),
        input_names=list(dummy.keys()),
        output_names=["logits", "insole_config", "measurements_hat"],
        dynamic_axes={k: {0: "B"} for k in dummy.keys()},
        opset_version=opset,
    )
    logger.info("Exported -> {} ({} bytes)", output, output.stat().st_size)


if __name__ == "__main__":
    app()
