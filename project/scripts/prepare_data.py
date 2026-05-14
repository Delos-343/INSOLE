"""
CLI: scan the data folder and emit a manifest CSV.

This is purely informational — it doesn't move files. Run it once after
syncing the Drive folder locally to verify that view-detection and
patient-ID extraction work on your filesystem.

    python scripts/prepare_data.py --data-dir data
"""

from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from backend.model.data.dataset import FootClassificationDataset

app = typer.Typer(add_completion=False, help="Inspect / index the data folder.")
console = Console()


@app.command()
def main(
    data_dir: Path = typer.Option("data", help="Root data folder."),
    output_csv: Path | None = typer.Option(
        None, help="Optionally write per-sample manifest to this CSV."
    ),
) -> None:
    ds = FootClassificationDataset(data_dir=data_dir, transform=None)
    console.print(f"Discovered [bold]{len(ds)}[/bold] samples in {data_dir.resolve()}")

    t = Table(title="Class distribution")
    t.add_column("Class")
    t.add_column("Count", justify="right")
    for cls, n in ds.class_counts().items():
        t.add_row(cls, str(n))
    console.print(t)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "patient_id", "label", "label_idx",
                "lateral_path", "top_path", "back_path",
                "calcaneal_inclination_deg", "heel_angle_deg",
                "arch_height_cm", "kite_angle_deg",
                "first_metatarsal_talus_deg", "measurement_mask",
            ])
            for s in ds.samples:
                w.writerow([
                    s.patient_id, s.label, s.label_idx,
                    s.lateral_path or "", s.top_path or "", s.back_path or "",
                    *map(float, s.measurements),
                    s.measurement_mask,
                ])
        console.print(f"[green]Manifest written to {output_csv}[/green]")


if __name__ == "__main__":
    app()
