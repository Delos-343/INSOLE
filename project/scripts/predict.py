"""
CLI: run inference on a triplet of images.

    python scripts/predict.py \\
        --lateral data/Heel/H1-99/P1097/lat.jpg \\
        --top     data/Heel/H1-99/P1097/top.jpg \\
        --back    data/Heel/H1-99/P1097/back.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.model.config import InferenceConfig
from backend.model.inference.predictor import Predictor

app = typer.Typer(add_completion=False, help="Predict foot class from images.")
console = Console()


@app.command()
def main(
    lateral: Path | None = typer.Option(None, help="Lateral / side view image."),
    top:     Path | None = typer.Option(None, help="Top / AP view image."),
    back:    Path | None = typer.Option(None, help="Back / posterior view image."),
    checkpoint: Path = typer.Option(
        "backend/model/checkpoints/best.pt", help="Trained checkpoint."
    ),
    arch_height_cm:             float | None = typer.Option(None, help="Optional measurement."),
    heel_angle_deg:             float | None = typer.Option(None, help="Optional measurement."),
    calcaneal_inclination_deg:  float | None = typer.Option(None, help="Optional measurement."),
    kite_angle_deg:             float | None = typer.Option(None, help="Optional measurement."),
    first_metatarsal_talus_deg: float | None = typer.Option(None, help="Optional measurement."),
    output_json: Path | None = typer.Option(None, help="Save full result to JSON."),
) -> None:
    if not any([lateral, top, back]):
        console.print("[red]At least one image must be provided.[/red]")
        raise typer.Exit(1)

    measurements = {
        k: v for k, v in {
            "arch_height_cm": arch_height_cm,
            "heel_angle_deg": heel_angle_deg,
            "calcaneal_inclination_deg": calcaneal_inclination_deg,
            "kite_angle_deg": kite_angle_deg,
            "first_metatarsal_talus_deg": first_metatarsal_talus_deg,
        }.items()
        if v is not None
    } or None

    cfg = InferenceConfig(checkpoint_path=checkpoint)
    predictor = Predictor(cfg)
    result = predictor.predict(lateral, top, back, measurements)

    # Headline panel.
    console.print(
        Panel.fit(
            f"[bold]{result.predicted_class}[/bold]\n"
            f"Confidence: [cyan]{result.confidence:.1%}[/cyan]   "
            f"Severity: [yellow]{result.severity_band}[/yellow]",
            title="Prediction",
        )
    )

    # Probabilities table.
    t = Table(title="Class probabilities")
    t.add_column("Class", style="bold")
    t.add_column("P", justify="right")
    for cls, p in result.class_probabilities.items():
        t.add_row(cls, f"{p:.3f}")
    console.print(t)

    # Measurements.
    mt = Table(title="Measurements used")
    mt.add_column("Name")
    mt.add_column("Value", justify="right")
    for k, v in result.measurements_predicted.items():
        mt.add_row(k, f"{v:.2f}")
    console.print(mt)

    if result.notes:
        console.print("[yellow]Notes:[/yellow]")
        for n in result.notes:
            console.print(f"  • {n}")

    if output_json:
        output_json.write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"[green]Wrote {output_json}[/green]")


if __name__ == "__main__":
    app()
