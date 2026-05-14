"""
Training worker QThread.

Strategy: API-first with local fallback.

  1. POST /api/training/runs to the backend container.
  2. If successful, poll GET /api/training/runs/{id} every 2 seconds.
     For each newly-appended history entry, emit `epoch_done`.
  3. On terminal status (COMPLETED / FAILED / CANCELLED), emit final signal.
  4. If the API is unreachable, transparently fall back to running the
     in-process Trainer on the user's machine — same behaviour as before.

This mirrors the inference worker's "remote-then-local" pattern and keeps
the GUI working whether or not the user has the Docker stack running.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import httpx
from PySide6.QtCore import QThread, Signal

from app.config import APP_CONFIG


class TrainingWorker(QThread):
    log = Signal(str)
    epoch_done = Signal(dict)
    step = Signal(dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        image_size: int,
        use_augmentation: bool,
        use_generative_branch: bool,
        api_base_url: str | None = None,
        use_local_fallback: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.use_generative_branch = use_generative_branch
        self.api_base_url = (api_base_url or APP_CONFIG.api_base_url).rstrip("/")
        self.use_local_fallback = use_local_fallback
        self._cancel_requested = False

    # ----------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self._run_remote()
            return
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            short = str(exc).split("\n", 1)[0][:120]
            if not self.use_local_fallback:
                self.failed.emit(f"Remote training unavailable: {short}")
                return
            self.log.emit(f"\n⚠ Backend container unreachable ({short}); falling back to local training.\n")

        # Local fallback path.
        try:
            self._run_local()
        except Exception as exc:
            self.log.emit(f"\nLocal training failed: {exc}\n{traceback.format_exc(limit=5)}")
            self.failed.emit(str(exc))

    # ============================================================= REMOTE
    def _run_remote(self) -> None:
        """POST to the backend container and poll for live progress."""
        # NOTE: we deliberately *do not* send the Windows host path — the
        # backend container has the dataset mounted at /workspace/data.
        # Leaving data_dir=None makes the route default to "data" (which,
        # combined with WORKDIR=/workspace, resolves correctly).
        payload = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "use_augmentation": self.use_augmentation,
            "use_generative_branch": self.use_generative_branch,
            "train_split": 0.8,
            "val_split": 0.1,
        }

        self.log.emit("Connecting to backend...")
        self.log.emit(f"  API: {self.api_base_url}")

        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{self.api_base_url}/api/training/runs", json=payload)
            r.raise_for_status()
            run_id = r.json()["id"]

        self.log.emit(f"✓ Started remote training run: {run_id}")
        self.log.emit(f"  batch_size     = {self.batch_size}")
        self.log.emit(f"  num_epochs     = {self.num_epochs}")
        self.log.emit(f"  learning_rate  = {self.learning_rate}")
        self.log.emit(f"  image_size     = {self.image_size}")
        self.log.emit("")
        self.log.emit("Polling for progress every 2s...")

        self._poll_until_done(run_id)

    def _poll_until_done(self, run_id: str) -> None:
        last_epoch_count = 0
        t0 = time.time()
        terminal = {"COMPLETED", "FAILED", "CANCELLED"}

        while not self._cancel_requested:
            time.sleep(2.0)
            try:
                with httpx.Client(timeout=10.0) as client:
                    r = client.get(f"{self.api_base_url}/api/training/runs/{run_id}")
                    r.raise_for_status()
                    status = r.json()
            except (httpx.HTTPError, ConnectionError) as exc:
                # transient network blip — keep trying for a bit
                self.log.emit(f"  (poll error: {exc!s:.80})")
                continue

            # Emit any new epochs.
            history = status.get("history") or []
            for i in range(last_epoch_count, len(history)):
                m = history[i]
                self.log.emit(
                    f"Epoch {m.get('epoch', i+1)}/{self.num_epochs}  "
                    f"loss={m.get('loss', 0):.4f}  "
                    f"val_acc={m.get('val_accuracy', 0):.4f}  "
                    f"val_f1={m.get('val_macro_f1', 0):.4f}"
                )
                self.epoch_done.emit(
                    {
                        "epoch": m.get("epoch", i + 1),
                        "total_epochs": self.num_epochs,
                        "metrics": m,
                        "elapsed_sec": time.time() - t0,
                    }
                )
            last_epoch_count = len(history)

            run_status = status.get("status")
            if run_status in terminal:
                if run_status == "COMPLETED":
                    self.log.emit("\n✓ Training complete.")
                    self.log.emit(
                        f"  best_val_accuracy = {status.get('best_val_accuracy', 0):.4f}"
                    )
                    self.log.emit(
                        f"  test_accuracy     = {status.get('test_accuracy', 0):.4f}"
                    )
                    self.log.emit(f"  checkpoint        = {status.get('checkpoint_path')}")
                    self.finished_ok.emit(
                        {
                            "best_val_accuracy": status.get("best_val_accuracy") or 0.0,
                            "test_metrics": {
                                "accuracy": status.get("test_accuracy") or 0.0,
                                "macro_f1": status.get("macro_f1") or 0.0,
                            },
                            "history": history,
                            "checkpoint_path": status.get("checkpoint_path"),
                        }
                    )
                else:
                    self.failed.emit(f"Remote run {run_status.lower()}.")
                return

    def cancel(self) -> None:
        """Cooperative cancellation flag."""
        self._cancel_requested = True

    # ============================================================== LOCAL
    def _run_local(self) -> None:
        """Original in-process trainer path (unchanged behaviour)."""
        from backend.model.config import ModelConfig, TrainingConfig
        from backend.model.training.trainer import Trainer

        train_cfg = TrainingConfig(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            image_size=self.image_size,
            use_augmentation=self.use_augmentation,
        )
        model_cfg = ModelConfig(use_generative_branch=self.use_generative_branch)

        def cb(payload: dict) -> None:
            t = payload.get("type")
            if t == "step":
                self.step.emit(payload)
                self.log.emit(
                    f"step {payload['step']:>5}  loss={payload['loss']:.4f}  "
                    f"running_acc={payload['running_acc']:.4f}"
                )
            elif t == "epoch_end":
                self.epoch_done.emit(payload)
                m = payload["metrics"]
                self.log.emit(
                    f"\nEpoch {payload['epoch']}/{payload['total_epochs']}  "
                    f"loss={m['loss']:.4f}  val_acc={m['val_accuracy']:.4f}  "
                    f"val_f1={m['val_macro_f1']:.4f}\n"
                )
            elif t == "done":
                self.log.emit(
                    f"Training complete. Best val_acc={payload['best_val_acc']:.4f}, "
                    f"test_acc={payload['test_metrics']['accuracy']:.4f}"
                )

        self.log.emit("Starting LOCAL training (Docker backend not reachable)...")
        self.log.emit(f"  data_dir       = {self.data_dir}")
        self.log.emit(f"  batch_size     = {self.batch_size}")
        self.log.emit(f"  num_epochs     = {self.num_epochs}")
        self.log.emit(f"  learning_rate  = {self.learning_rate}")
        self.log.emit(f"  image_size     = {self.image_size}")
        self.log.emit(f"  augmentation   = {self.use_augmentation}")
        self.log.emit(f"  generative VAE = {self.use_generative_branch}\n")

        trainer = Trainer(train_cfg, model_cfg, progress_cb=cb)
        result = trainer.fit()
        self.finished_ok.emit(result)