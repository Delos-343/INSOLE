"""
QThread wrapper around the inference call so the UI never blocks.

The worker tries the HTTP API first (if reachable); falls back to running
the in-process Predictor — which makes the desktop app fully functional
without the FastAPI service running.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import httpx
from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        api_base_url: str,
        lateral_path: str | None,
        top_path: str | None,
        back_path: str | None,
        measurements: dict,
        patient_code: str | None = None,
        use_local_fallback: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.api_base_url = api_base_url.rstrip("/")
        self.lateral_path = lateral_path
        self.top_path = top_path
        self.back_path = back_path
        self.measurements = measurements
        self.patient_code = patient_code
        self.use_local_fallback = use_local_fallback

    def run(self) -> None:
        try:
            result = self._try_api()
            self.finished_ok.emit(result)
        except Exception as api_exc:
            if not self.use_local_fallback:
                self.failed.emit(str(api_exc))
                return
            try:
                result = self._try_local()
                result["notes"] = list(result.get("notes", [])) + [
                    f"Used in-process model (API unreachable: {api_exc!s:.80})"
                ]
                self.finished_ok.emit(result)
            except Exception as local_exc:
                self.failed.emit(
                    f"Both API and local inference failed.\n\n"
                    f"API error: {api_exc}\n\n"
                    f"Local error:\n{traceback.format_exc(limit=4)}\n"
                    f"{local_exc}"
                )

    # ------------------------------------------------------------------ API
    def _try_api(self) -> dict:
        files: dict[str, tuple[str, bytes, str]] = {}
        for field, path in [
            ("lateral", self.lateral_path),
            ("top", self.top_path),
            ("back", self.back_path),
        ]:
            if path and Path(path).exists():
                files[field] = (
                    Path(path).name,
                    Path(path).read_bytes(),
                    "application/octet-stream",
                )

        data: dict[str, str] = {}
        if self.measurements:
            data["measurements_json"] = json.dumps(self.measurements)
        if self.patient_code:
            data["patient_code"] = self.patient_code

        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{self.api_base_url}/api/classify", files=files, data=data)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------ Local
    def _try_local(self) -> dict:
        # Import locally so the GUI doesn't pull torch on cold start.
        from backend.model.config import InferenceConfig
        from backend.model.inference.predictor import Predictor

        predictor = Predictor(InferenceConfig())
        t0 = time.perf_counter()
        result = predictor.predict(
            lateral_path=self.lateral_path,
            top_path=self.top_path,
            back_path=self.back_path,
            measurements=self.measurements or None,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        out = result.to_dict()
        out["inference_time_ms"] = elapsed_ms
        return out
