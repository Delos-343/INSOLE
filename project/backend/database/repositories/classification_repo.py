"""Repository pattern over the Classification ORM model."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Classification, FootClass, SeverityBand


class ClassificationRepository:
    """All read/write operations for the `Classification` table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------- C
    def create(self, **fields) -> Classification:
        # Translate display strings to enums where needed.
        if isinstance(fields.get("predicted_class"), str):
            fields["predicted_class"] = FootClass.from_display(fields["predicted_class"])
        if isinstance(fields.get("rule_based_label"), str):
            fields["rule_based_label"] = FootClass.from_display(fields["rule_based_label"])
        if isinstance(fields.get("severity_band"), str):
            fields["severity_band"] = SeverityBand(fields["severity_band"].upper())

        row = Classification(**fields)
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    # ------------------------------------------------------------------- R
    def get(self, classification_id: str) -> Classification | None:
        return self.db.get(Classification, classification_id)

    def list_recent(self, limit: int = 50) -> Sequence[Classification]:
        stmt = (
            select(Classification)
            .order_by(Classification.created_at.desc())
            .limit(limit)
        )
        return self.db.execute(stmt).scalars().all()

    def for_patient(self, patient_id: str) -> Sequence[Classification]:
        stmt = (
            select(Classification)
            .where(Classification.patient_id == patient_id)
            .order_by(Classification.created_at.desc())
        )
        return self.db.execute(stmt).scalars().all()

    # ------------------------------------------------------------------- D
    def delete(self, classification_id: str) -> bool:
        row = self.get(classification_id)
        if row is None:
            return False
        self.db.delete(row)
        self.db.commit()
        return True
