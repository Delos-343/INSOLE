"""
Seed the database with a few demo patients and a sham classification
record so the GUI has something to render before the user runs anything.

    python scripts/seed_demo_data.py
"""

from __future__ import annotations

import typer
from loguru import logger

from backend.database.connection import create_all, db_session
from backend.database.models import FootClass, SeverityBand
from backend.database.repositories.classification_repo import ClassificationRepository
from backend.database.repositories.patient_repo import PatientRepository

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    create_all()

    with db_session() as db:
        p_repo = PatientRepository(db)
        c_repo = ClassificationRepository(db)

        for code, age in [("P1097", 34), ("P1098", 41), ("P1099", 28)]:
            patient = p_repo.upsert_by_code(code, age=age)
            c_repo.create(
                patient_id=patient.id,
                predicted_class=FootClass.NORMAL_FOOT,
                confidence=0.93,
                class_probs={
                    "Severe Flat Arch": 0.01,
                    "Flat Arch":        0.03,
                    "Normal Foot":      0.93,
                    "High Arch":        0.02,
                    "Severe High Arch": 0.01,
                },
                severity_band=SeverityBand.NORMAL,
                rule_based_label=FootClass.NORMAL_FOOT,
                arch_height_cm=4.5,
                heel_angle_deg=2.0,
                measurements_were_provided=True,
                model_version="seed",
            )
        logger.info("Seeded demo data.")


if __name__ == "__main__":
    app()
