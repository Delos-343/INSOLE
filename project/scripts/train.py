"""
CLI: train a model from the terminal without spinning up the GUI.

    python scripts/train.py --data-dir data --epochs 50 --batch-size 16
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from backend.model.config import ModelConfig, TrainingConfig
from backend.model.training.trainer import Trainer

app = typer.Typer(add_completion=False, help="Train the foot classifier.")
console = Console()


@app.command()
def main(
    data_dir: Path = typer.Option("data", help="Root data folder."),
    epochs: int = typer.Option(50, help="Number of epochs."),
    batch_size: int = typer.Option(16, help="Training batch size."),
    learning_rate: float = typer.Option(1e-4, help="Initial learning rate."),
    image_size: int = typer.Option(256, help="Square image size."),
    output_dir: Path = typer.Option(
        "backend/model/checkpoints", help="Where checkpoints are written."
    ),
    no_aug: bool = typer.Option(False, help="Disable augmentation."),
    no_vae: bool = typer.Option(False, help="Disable generative VAE branch."),
    device: str = typer.Option("auto", help="cpu | cuda | mps | auto"),
) -> None:
    train_cfg = TrainingConfig(
        data_dir=data_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        output_dir=output_dir,
        use_augmentation=not no_aug,
        device=device,  # type: ignore[arg-type]
    )
    model_cfg = ModelConfig(use_generative_branch=not no_vae)

    # Pretty print configs.
    t = Table(title="Configuration", show_header=False)
    for k, v in {**train_cfg.__dict__, **model_cfg.__dict__}.items():
        t.add_row(str(k), str(v))
    console.print(t)

    trainer = Trainer(train_cfg, model_cfg)
    result = trainer.fit()

    logger.info("Best validation accuracy: {:.4f}", result["best_val_accuracy"])
    logger.info("Test metrics: {}", result["test_metrics"])


if __name__ == "__main__":
    app()
