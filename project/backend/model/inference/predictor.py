"""
Inference engine — MEASUREMENT-FIRST.

Rationale (proven by scripts/diagnose_model.py)
-----------------------------------------------
The five classes are *defined* by the project brief as arch-height-cm
bands. The diagnostic showed:
  - Rule on true measurements:           ~100% (deterministic; it IS the def)
  - Model with measurements present:      ~88%
  - Model images-only (no measurements):  ~33%  (arch height is not
                                                 visually recoverable from
                                                 the supplied photo views)

Therefore classification must NOT be a neural-network guess. It is a
deterministic lookup against the brief's bands. The neural network's
honest, value-adding role is to *estimate* the measurements from images
when a clinician has not measured them — clearly flagged as an estimate
with lower trust, never presented as authoritative truth.

Decision flow
-------------
    measurements provided?
        YES -> classify by exact cm rule        [source = MEASURED]
        NO  -> model estimates measurements
               from images, then SAME cm rule   [source = IMAGE_ESTIMATED]

The result object records which path was taken so the UI/clinician is
never misled about provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from backend.model.architectures.classifier import build_classifier
from backend.model.architectures.generative_vae import InsoleConfigHead
from backend.model.architectures.measurement_predictor import MeasurementHead
from backend.model.config import ARCH_HEIGHT_BANDS, CLASS_NAMES, InferenceConfig, ModelConfig
from backend.model.data.transforms import build_eval_transform
from backend.model.training.trainer import resolve_device
from backend.model.utils.checkpoint import find_latest_checkpoint, load_checkpoint


class NoTrainedModelError(RuntimeError):
    pass


class ClassificationSource(str, Enum):
    MEASURED = "measured"                 # authoritative — clinician measurements
    IMAGE_ESTIMATED = "image_estimated"   # assistive — model-estimated, lower trust


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict[str, float]
    measurements_used: dict[str, float]          # the values the rule classified on
    measurements_provided: dict[str, float | None]
    measurements_estimated: dict[str, float]     # always the model's estimate (for QA)
    classification_source: str                   # ClassificationSource value
    insole_configuration: dict[str, float]
    severity_band: str
    notes: list[str]
    checkpoint_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The deterministic classifier — this IS the brief's definition.
# ---------------------------------------------------------------------------
def classify_by_arch_height_cm(arch_height_cm: float) -> tuple[str, int]:
    for idx, name in enumerate(CLASS_NAMES):
        lo, hi = ARCH_HEIGHT_BANDS[name]
        if lo <= arch_height_cm < hi:
            return name, idx
    # Out of all bands (e.g. negative) -> clamp sensibly.
    if arch_height_cm < ARCH_HEIGHT_BANDS[CLASS_NAMES[0]][0]:
        return CLASS_NAMES[0], 0
    return CLASS_NAMES[-1], len(CLASS_NAMES) - 1


def _severity_band(class_name: str) -> str:
    return {
        "Normal Foot": "normal",
        "Flat Arch": "moderate",
        "High Arch": "moderate",
        "Severe Flat Arch": "severe",
        "Severe High Arch": "severe",
    }.get(class_name, "unknown")


class Predictor:
    """Stateful predictor — load once, call many times."""

    def __init__(self, cfg: InferenceConfig | None = None) -> None:
        self.cfg = cfg or InferenceConfig()
        self.device = resolve_device(self.cfg.device)
        self.transform = build_eval_transform(self.cfg.image_size)
        self.checkpoint_path: Path | None = None

        model_cfg, weights = self._load_weights()
        self.model_cfg = model_cfg
        self.model = build_classifier(model_cfg).to(self.device)
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        n_loaded = len(weights) - len(unexpected)
        if n_loaded == 0:
            raise NoTrainedModelError(
                "Checkpoint loaded but ZERO weights bound — architecture mismatch."
            )
        logger.info(
            "Predictor ready — {} weights from {} (measurement-first mode)",
            n_loaded,
            self.checkpoint_path,
        )
        self.model.eval()

    def _load_weights(self) -> tuple[ModelConfig, dict]:
        explicit = Path(self.cfg.checkpoint_path)
        chosen = explicit if explicit.exists() else find_latest_checkpoint(explicit.parent)
        if chosen is None:
            raise NoTrainedModelError(
                f"No trained checkpoint found in {explicit.parent.resolve()}. Train first."
            )
        self.checkpoint_path = chosen
        try:
            state, meta = load_checkpoint(chosen, map_location=self.device)
        except Exception as exc:
            raise NoTrainedModelError(f"Failed to load {chosen}: {exc}") from exc
        model_cfg = ModelConfig(**meta["model_cfg"]) if meta.get("model_cfg") else ModelConfig()
        return model_cfg, state

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def predict(
        self,
        lateral_path: str | Path | None,
        top_path: str | Path | None,
        back_path: str | Path | None,
        measurements: Mapping[str, float] | None = None,
    ) -> PredictionResult:
        notes: list[str] = []

        lat_t, lat_present = self._prep_image(lateral_path)
        top_t, top_present = self._prep_image(top_path)
        bak_t, bak_present = self._prep_image(back_path)
        for present, name in [(lat_present, "lateral"), (top_present, "top"),
                              (bak_present, "back")]:
            if not present:
                notes.append(f"No {name} view supplied.")

        meas_vec, mask = self._prep_measurements(measurements)
        provided = {
            n: (measurements.get(n) if measurements else None)
            for n in MeasurementHead.MEASUREMENT_NAMES
        }

        # Always run the model — we need its measurement ESTIMATE for QA,
        # the insole-config head, and the image-only fallback path.
        outputs = self.model(
            lateral=lat_t.unsqueeze(0).to(self.device),
            top=top_t.unsqueeze(0).to(self.device),
            back=bak_t.unsqueeze(0).to(self.device),
            measurements=meas_vec.unsqueeze(0).to(self.device),
            measurement_mask=torch.tensor([[mask]], device=self.device),
        )
        est = outputs["measurements_hat"].squeeze(0).cpu().numpy()
        estimated = {
            n: float(est[i]) for i, n in enumerate(MeasurementHead.MEASUREMENT_NAMES)
        }

        # -------- Decide the classification path --------
        provided_arch = (
            measurements.get("arch_height_cm") if measurements else None
        )
        has_real_arch = provided_arch is not None and float(provided_arch) > 0

        if has_real_arch:
            arch_cm = float(provided_arch)
            source = ClassificationSource.MEASURED
            used = {
                n: (float(measurements[n]) if measurements and measurements.get(n) is not None
                    else estimated[n])
                for n in MeasurementHead.MEASUREMENT_NAMES
            }
            notes.append(
                "Classified deterministically from the PROVIDED arch height "
                f"({arch_cm:.2f} cm) using the project's clinical bands. "
                "This result is authoritative."
            )
        else:
            arch_cm = float(estimated["arch_height_cm"])
            source = ClassificationSource.IMAGE_ESTIMATED
            used = dict(estimated)
            notes.append(
                f"No arch-height measurement supplied. Arch height was "
                f"ESTIMATED from the images ({arch_cm:.2f} cm) and is "
                f"approximate. Diagnostic testing shows image-only estimates "
                f"are unreliable for this dataset — treat as assistive only "
                f"and confirm with a clinical measurement."
            )

        pred_class, pred_idx = classify_by_arch_height_cm(arch_cm)

        # Confidence: for MEASURED it is definitionally certain (1.0). For
        # ESTIMATED, derive a soft confidence from how far the estimate sits
        # from the nearest band boundary (closer to a boundary = less sure).
        if source is ClassificationSource.MEASURED:
            confidence = 1.0
            probs = {n: (1.0 if i == pred_idx else 0.0)
                     for i, n in enumerate(CLASS_NAMES)}
        else:
            confidence, probs = self._estimate_confidence(arch_cm, pred_idx)
            notes.append(
                "Estimated-path confidence reflects distance from band "
                "edges, not measured certainty."
            )

        insole = outputs["insole_config"].squeeze(0).cpu().numpy()
        insole_dict = {
            n: float(insole[i]) for i, n in enumerate(InsoleConfigHead.OUTPUT_NAMES)
        }

        return PredictionResult(
            predicted_class=pred_class,
            predicted_class_idx=pred_idx,
            confidence=confidence,
            class_probabilities=probs,
            measurements_used=used,
            measurements_provided=provided,
            measurements_estimated=estimated,
            classification_source=source.value,
            insole_configuration=insole_dict,
            severity_band=_severity_band(pred_class),
            notes=notes,
            checkpoint_used=str(self.checkpoint_path),
        )

    # ------------------------------------------------------------- helpers
    def _estimate_confidence(
        self, arch_cm: float, pred_idx: int
    ) -> tuple[float, dict[str, float]]:
        """Soft confidence for the estimated path.

        We place a small Gaussian around the estimated arch height and
        integrate its mass into each band. Width sigma reflects the
        empirically poor image-only reliability, so estimated-path
        confidence stays honestly modest.
        """
        sigma = 0.6  # cm — deliberately wide; image-only is unreliable
        edges = []
        for name in CLASS_NAMES:
            lo, hi = ARCH_HEIGHT_BANDS[name]
            edges.append((name, lo, hi))

        def _norm_cdf(x: float) -> float:
            # standard normal CDF via erf
            import math
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        mass = {}
        for name, lo, hi in edges:
            lo_z = (lo - arch_cm) / sigma
            hi_z = (hi - arch_cm) / sigma
            mass[name] = max(0.0, _norm_cdf(hi_z) - _norm_cdf(lo_z))
        total = sum(mass.values()) or 1.0
        probs = {k: v / total for k, v in mass.items()}
        return float(probs[CLASS_NAMES[pred_idx]]), probs

    def _prep_image(self, path: str | Path | None) -> tuple[torch.Tensor, bool]:
        if path is None or not Path(path).exists():
            black = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), (0, 0, 0))
            return self.transform(black), False
        return self.transform(Image.open(path).convert("RGB")), True

    def _prep_measurements(self, m: Mapping[str, float] | None) -> tuple[torch.Tensor, float]:
        vec = torch.zeros(5, dtype=torch.float32)
        if not m:
            return vec, 0.0
        present = False
        for i, name in enumerate(MeasurementHead.MEASUREMENT_NAMES):
            if name in m and m[name] is not None:
                vec[i] = float(m[name])
                present = True
        return vec, (1.0 if present else 0.0)


def predict(
    lateral_path: str | Path | None,
    top_path: str | Path | None,
    back_path: str | Path | None,
    measurements: Mapping[str, float] | None = None,
    cfg: InferenceConfig | None = None,
) -> PredictionResult:
    return Predictor(cfg).predict(lateral_path, top_path, back_path, measurements)
