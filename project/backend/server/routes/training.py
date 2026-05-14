"""
Training endpoints — start a run, query status (with live progress), list history.

Training is launched in a background thread so the HTTP request returns
immediately. A progress callback writes each epoch's metrics into the
TrainingRun.history column so the GUI can poll for live updates.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from backend.database.connection import db_session, get_db
from backend.database.models import TrainingRun
from backend.database.repositories.training_run_repo import TrainingRunRepository
from backend.database.schemas import TrainingRequest, TrainingStatusOut
from backend.model.config import ModelConfig, TrainingConfig

router = APIRouter()

# Track active background trainers so we can prevent duplicates.
_active_runs: dict[str, threading.Thread] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _config_to_jsonable(cfg: Any) -> dict:
    """Convert a dataclass (or any object with __dict__) to a JSON-safe dict.

    Dataclass fields like ``data_dir: Path`` are not natively JSON-encodable;
    PostgreSQL's JSON column rejects them. We coerce non-primitive values to
    strings so they round-trip cleanly through the JSON column.
    """
    out: dict = {}
    for k, v in cfg.__dict__.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v  # assume already JSON-safe
        else:
            out[k] = str(v)  # Path, Literal, Enum, etc.
    return out


# ---------------------------------------------------------------------------
# Background trainer thread
# ---------------------------------------------------------------------------
def _run_training_thread(run_id: str, training_cfg: TrainingConfig, model_cfg: ModelConfig) -> None:
    """Run the trainer; update DB row with progress + outcome."""
    from backend.model.training.trainer import Trainer  # local import keeps cold-start cheap

    with db_session() as db:
        TrainingRunRepository(db).mark_running(run_id)

    # ----- progress callback -----
    # Every epoch_end, append the metrics to the row's `history` JSON
    # column. The GUI polls GET /api/training/runs/{id} every couple of
    # seconds and emits new entries as `epoch_done` signals.
    def progress_cb(payload: dict) -> None:
        if payload.get("type") != "epoch_end":
            return
        try:
            with db_session() as db:
                row = db.get(TrainingRun, run_id)
                if row is None:
                    return
                history = list(row.history or [])
                history.append(payload["metrics"])
                row.history = history
                row.total_epochs = len(history)
                val_acc = payload["metrics"].get("val_accuracy")
                if val_acc is not None and (
                    row.best_val_accuracy is None or val_acc > row.best_val_accuracy
                ):
                    row.best_val_accuracy = float(val_acc)
                db.commit()
        except Exception as exc:
            logger.warning("Progress callback DB write failed: {}", exc)

    try:
        trainer = Trainer(training_cfg, model_cfg, progress_cb=progress_cb)
        result = trainer.fit()

        # Move best.pt -> stable path keyed by run id.
        out_dir = Path(training_cfg.output_dir)
        best_src = out_dir / "best.pt"
        best_dst = out_dir / f"run_{run_id}.pt"
        if best_src.exists():
            try:
                best_src.replace(best_dst)
            except Exception:
                pass

        with db_session() as db:
            TrainingRunRepository(db).mark_completed(
                run_id,
                best_val_accuracy=result["best_val_accuracy"],
                test_accuracy=result["test_metrics"].get("accuracy"),
                macro_f1=result["test_metrics"].get("macro_f1"),
                total_epochs=len(result["history"]),
                history=result["history"],
                checkpoint_path=str(best_dst if best_dst.exists() else best_src),
                model_version=f"v0.1.0+{run_id[:8]}",
            )
        logger.info("Training run {} completed.", run_id)

    except Exception:
        logger.exception("Training run {} failed", run_id)
        with db_session() as db:
            TrainingRunRepository(db).mark_failed(run_id)
    finally:
        _active_runs.pop(run_id, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/runs", response_model=TrainingStatusOut, status_code=202)
async def start_training(
    req: TrainingRequest,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingStatusOut:
    # Build configs from the request.
    training_cfg = TrainingConfig(
        data_dir=Path(req.data_dir) if req.data_dir else Path("data"),
        batch_size=req.batch_size,
        num_epochs=req.num_epochs,
        learning_rate=req.learning_rate,
        image_size=req.image_size,
        use_augmentation=req.use_augmentation,
        train_split=req.train_split,
        val_split=req.val_split,
    )
    model_cfg = ModelConfig(use_generative_branch=req.use_generative_branch)

    # IMPORTANT: serialise to JSON-safe dicts before writing the row.
    repo = TrainingRunRepository(db)
    row = repo.create(
        name=req.name,
        training_config=_config_to_jsonable(training_cfg),
        model_config=_config_to_jsonable(model_cfg),
    )

    # Kick off background thread.
    t = threading.Thread(
        target=_run_training_thread, args=(row.id, training_cfg, model_cfg), daemon=True
    )
    _active_runs[row.id] = t
    t.start()

    return _row_to_status(row)


@router.get("/runs", response_model=list[TrainingStatusOut])
async def list_runs(limit: int = 20, db: Session = Depends(get_db)) -> Sequence[TrainingStatusOut]:
    rows = TrainingRunRepository(db).list_recent(limit)
    return [_row_to_status(r) for r in rows]


@router.get("/runs/{run_id}", response_model=TrainingStatusOut)
async def get_run(run_id: str, db: Session = Depends(get_db)) -> TrainingStatusOut:
    row = TrainingRunRepository(db).get(run_id)
    if row is None:
        raise HTTPException(404, detail="Training run not found.")
    return _row_to_status(row)


def _row_to_status(row) -> TrainingStatusOut:
    return TrainingStatusOut(
        id=row.id,
        name=row.name,
        status=row.status,
        best_val_accuracy=row.best_val_accuracy,
        test_accuracy=row.test_accuracy,
        macro_f1=row.macro_f1,
        total_epochs=row.total_epochs,
        trained_minutes=row.trained_minutes,
        started_at=row.started_at,
        finished_at=row.finished_at,
        checkpoint_path=row.checkpoint_path,
        history=row.history,
    )