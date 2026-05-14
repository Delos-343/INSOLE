"""Repository for TrainingRun records."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import TrainingRun, TrainStatus


class TrainingRunRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, name: str | None, training_config: dict, model_config: dict) -> TrainingRun:
        row = TrainingRun(
            name=name,
            training_config=training_config,
            model_config_=model_config,
            status=TrainStatus.QUEUED,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def mark_running(self, run_id: str) -> None:
        row = self.db.get(TrainingRun, run_id)
        if row is None:
            return
        row.status = TrainStatus.RUNNING
        row.started_at = datetime.utcnow()
        self.db.commit()

    def mark_completed(self, run_id: str, **metrics) -> None:
        row = self.db.get(TrainingRun, run_id)
        if row is None:
            return
        row.status = TrainStatus.COMPLETED
        row.finished_at = datetime.utcnow()
        for k, v in metrics.items():
            if hasattr(row, k):
                setattr(row, k, v)
        if row.started_at:
            row.trained_minutes = int((row.finished_at - row.started_at).total_seconds() / 60)
        self.db.commit()

    def mark_failed(self, run_id: str) -> None:
        row = self.db.get(TrainingRun, run_id)
        if row is None:
            return
        row.status = TrainStatus.FAILED
        row.finished_at = datetime.utcnow()
        self.db.commit()

    def get(self, run_id: str) -> TrainingRun | None:
        return self.db.get(TrainingRun, run_id)

    def list_recent(self, limit: int = 20):
        return self.db.execute(
            select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit)
        ).scalars().all()
