"""
SQLAlchemy ORM models — mirror of the Prisma schema.

Why both?
---------
The Prisma schema is the design document; SQLAlchemy is what actually runs
inside the FastAPI service (because Prisma's Python client is still maturing).
Whenever the schema changes, update BOTH files and regenerate the Alembic
migration.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Sex(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class FootClass(str, enum.Enum):
    SEVERE_FLAT_ARCH = "SEVERE_FLAT_ARCH"
    FLAT_ARCH = "FLAT_ARCH"
    NORMAL_FOOT = "NORMAL_FOOT"
    HIGH_ARCH = "HIGH_ARCH"
    SEVERE_HIGH_ARCH = "SEVERE_HIGH_ARCH"

    @classmethod
    def from_display(cls, name: str) -> "FootClass":
        return cls[name.upper().replace(" ", "_")]


class SeverityBand(str, enum.Enum):
    NORMAL = "NORMAL"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    UNKNOWN = "UNKNOWN"


class MeasurementSource(str, enum.Enum):
    CLINICIAN = "CLINICIAN"
    IMAGE_AUTO = "IMAGE_AUTO"
    IMPORT_XLSX = "IMPORT_XLSX"
    X_RAY = "X_RAY"


class TrainStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[Sex | None] = mapped_column(Enum(Sex), nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    classifications: Mapped[list["Classification"]] = relationship(back_populates="patient")
    measurements: Mapped[list["Measurement"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    patient_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="SET NULL"), nullable=True
    )
    patient: Mapped["Patient | None"] = relationship(back_populates="classifications")

    predicted_class: Mapped[FootClass] = mapped_column(Enum(FootClass))
    confidence: Mapped[float] = mapped_column(Float)
    class_probs: Mapped[dict] = mapped_column(JSON)
    severity_band: Mapped[SeverityBand] = mapped_column(Enum(SeverityBand))
    rule_based_label: Mapped[FootClass | None] = mapped_column(Enum(FootClass), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    lateral_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    top_image_path:     Mapped[str | None] = mapped_column(String(512), nullable=True)
    back_image_path:    Mapped[str | None] = mapped_column(String(512), nullable=True)

    calcaneal_inclination_deg:  Mapped[float | None] = mapped_column(Float, nullable=True)
    heel_angle_deg:             Mapped[float | None] = mapped_column(Float, nullable=True)
    arch_height_cm:             Mapped[float | None] = mapped_column(Float, nullable=True)
    kite_angle_deg:             Mapped[float | None] = mapped_column(Float, nullable=True)
    first_metatarsal_talus_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    measurements_were_provided: Mapped[bool] = mapped_column(Boolean, default=False)

    arch_support_height:    Mapped[float | None] = mapped_column(Float, nullable=True)
    heel_cup_depth:         Mapped[float | None] = mapped_column(Float, nullable=True)
    medial_post_strength:   Mapped[float | None] = mapped_column(Float, nullable=True)
    lateral_wedge_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    forefoot_cushioning:    Mapped[float | None] = mapped_column(Float, nullable=True)

    model_version:     Mapped[str | None] = mapped_column(String(64), nullable=True)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
class Measurement(Base):
    __tablename__ = "measurements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="CASCADE")
    )
    patient: Mapped["Patient"] = relationship(back_populates="measurements")

    calcaneal_inclination_deg:  Mapped[float | None] = mapped_column(Float, nullable=True)
    heel_angle_deg:             Mapped[float | None] = mapped_column(Float, nullable=True)
    arch_height_cm:             Mapped[float | None] = mapped_column(Float, nullable=True)
    kite_angle_deg:             Mapped[float | None] = mapped_column(Float, nullable=True)
    first_metatarsal_talus_deg: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[MeasurementSource] = mapped_column(
        Enum(MeasurementSource), default=MeasurementSource.CLINICIAN
    )
    taken_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[TrainStatus] = mapped_column(Enum(TrainStatus), default=TrainStatus.QUEUED)

    training_config: Mapped[dict] = mapped_column(JSON)
    model_config_:   Mapped[dict] = mapped_column("model_config", JSON)  # quoted to avoid pydantic clash

    best_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_accuracy:     Mapped[float | None] = mapped_column(Float, nullable=True)
    macro_f1:          Mapped[float | None] = mapped_column(Float, nullable=True)
    total_epochs:      Mapped[int | None]   = mapped_column(Integer, nullable=True)
    trained_minutes:   Mapped[int | None]   = mapped_column(Integer, nullable=True)
    num_samples_train: Mapped[int | None]   = mapped_column(Integer, nullable=True)
    num_samples_val:   Mapped[int | None]   = mapped_column(Integer, nullable=True)
    num_samples_test:  Mapped[int | None]   = mapped_column(Integer, nullable=True)

    history: Mapped[list | None] = mapped_column(JSON, nullable=True)

    checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_version:   Mapped[str | None] = mapped_column(String(64), nullable=True)

    started_at:  Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at:  Mapped[datetime]        = mapped_column(DateTime, default=datetime.utcnow)
