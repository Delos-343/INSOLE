"""Initial schema — patients, classifications, measurements, training_runs.

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-14
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


sex_enum = sa.Enum("MALE", "FEMALE", "OTHER", "UNKNOWN", name="sex")
foot_class_enum = sa.Enum(
    "SEVERE_FLAT_ARCH", "FLAT_ARCH", "NORMAL_FOOT", "HIGH_ARCH", "SEVERE_HIGH_ARCH",
    name="footclass",
)
severity_enum = sa.Enum("NORMAL", "MODERATE", "SEVERE", "UNKNOWN", name="severityband")
meas_source_enum = sa.Enum(
    "CLINICIAN", "IMAGE_AUTO", "IMPORT_XLSX", "X_RAY", name="measurementsource"
)
train_status_enum = sa.Enum(
    "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", name="trainstatus"
)


def upgrade() -> None:
    bind = op.get_bind()
    for e in (sex_enum, foot_class_enum, severity_enum, meas_source_enum, train_status_enum):
        e.create(bind, checkfirst=True)

    op.create_table(
        "patients",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("code", sa.String(16), nullable=False, unique=True),
        sa.Column("age", sa.Integer()),
        sa.Column("sex", sex_enum),
        sa.Column("height_cm", sa.Float()),
        sa.Column("weight_kg", sa.Float()),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_patients_code", "patients", ["code"], unique=True)

    op.create_table(
        "classifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), sa.ForeignKey("patients.id", ondelete="SET NULL")),
        sa.Column("predicted_class", foot_class_enum, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("class_probs", sa.JSON(), nullable=False),
        sa.Column("severity_band", severity_enum, nullable=False),
        sa.Column("rule_based_label", foot_class_enum),
        sa.Column("notes", sa.Text()),
        sa.Column("lateral_image_path", sa.String(512)),
        sa.Column("top_image_path", sa.String(512)),
        sa.Column("back_image_path", sa.String(512)),
        sa.Column("calcaneal_inclination_deg", sa.Float()),
        sa.Column("heel_angle_deg", sa.Float()),
        sa.Column("arch_height_cm", sa.Float()),
        sa.Column("kite_angle_deg", sa.Float()),
        sa.Column("first_metatarsal_talus_deg", sa.Float()),
        sa.Column("measurements_were_provided", sa.Boolean(), default=False),
        sa.Column("arch_support_height", sa.Float()),
        sa.Column("heel_cup_depth", sa.Float()),
        sa.Column("medial_post_strength", sa.Float()),
        sa.Column("lateral_wedge_strength", sa.Float()),
        sa.Column("forefoot_cushioning", sa.Float()),
        sa.Column("model_version", sa.String(64)),
        sa.Column("inference_time_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "measurements",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36),
                  sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
        sa.Column("calcaneal_inclination_deg", sa.Float()),
        sa.Column("heel_angle_deg", sa.Float()),
        sa.Column("arch_height_cm", sa.Float()),
        sa.Column("kite_angle_deg", sa.Float()),
        sa.Column("first_metatarsal_talus_deg", sa.Float()),
        sa.Column("source", meas_source_enum, server_default="CLINICIAN", nullable=False),
        sa.Column("taken_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("notes", sa.Text()),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128)),
        sa.Column("status", train_status_enum, server_default="QUEUED", nullable=False),
        sa.Column("training_config", sa.JSON(), nullable=False),
        sa.Column("model_config", sa.JSON(), nullable=False),
        sa.Column("best_val_accuracy", sa.Float()),
        sa.Column("test_accuracy", sa.Float()),
        sa.Column("macro_f1", sa.Float()),
        sa.Column("total_epochs", sa.Integer()),
        sa.Column("trained_minutes", sa.Integer()),
        sa.Column("num_samples_train", sa.Integer()),
        sa.Column("num_samples_val", sa.Integer()),
        sa.Column("num_samples_test", sa.Integer()),
        sa.Column("history", sa.JSON()),
        sa.Column("checkpoint_path", sa.String(512)),
        sa.Column("model_version", sa.String(64)),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("finished_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
    op.drop_table("measurements")
    op.drop_table("classifications")
    op.drop_index("ix_patients_code", table_name="patients")
    op.drop_table("patients")
    bind = op.get_bind()
    for e in (train_status_enum, meas_source_enum, severity_enum, foot_class_enum, sex_enum):
        e.drop(bind, checkfirst=True)
