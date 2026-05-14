"""Repository for Patient records."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Patient


class PatientRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_by_code(self, code: str, **fields) -> Patient:
        """Create the patient by code or return the existing one (updating fields)."""
        row = self.db.execute(
            select(Patient).where(Patient.code == code.upper())
        ).scalar_one_or_none()
        if row is None:
            row = Patient(code=code.upper(), **fields)
            self.db.add(row)
        else:
            for k, v in fields.items():
                if v is not None:
                    setattr(row, k, v)
        self.db.commit()
        self.db.refresh(row)
        return row

    def get(self, patient_id: str) -> Patient | None:
        return self.db.get(Patient, patient_id)

    def get_by_code(self, code: str) -> Patient | None:
        return self.db.execute(
            select(Patient).where(Patient.code == code.upper())
        ).scalar_one_or_none()

    def list_all(self, limit: int = 100):
        return self.db.execute(select(Patient).limit(limit)).scalars().all()
