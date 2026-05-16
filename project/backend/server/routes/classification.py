"""
POST /api/classify — measurement-first inference endpoint.

Accepts three optional image uploads + an optional measurements blob,
runs the measurement-first Predictor, persists the result, returns
ClassificationOut (now including the classification source so callers
know whether the result is authoritative or an image estimate).
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.repositories.classification_repo import ClassificationRepository
from backend.database.repositories.patient_repo import PatientRepository
from backend.database.schemas import ClassificationOut, InsoleConfigOut, MeasurementsIn
from backend.server.dependencies import get_predictor
from backend.server.utils.file_handler import save_upload

router = APIRouter()


@router.post("/classify", response_model=ClassificationOut)
async def classify(
    lateral: UploadFile | None = File(None),
    top: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    measurements_json: str | None = Form(None),
    patient_code: str | None = Form(None),
    predictor=Depends(get_predictor),
    db: Session = Depends(get_db),
) -> ClassificationOut:
    if lateral is None and top is None and back is None and not measurements_json:
        raise HTTPException(
            400, detail="Provide at least one image or a measurements payload."
        )

    lat_path = await save_upload(lateral, "lat") if lateral else None
    top_path = await save_upload(top, "top") if top else None
    bak_path = await save_upload(back, "back") if back else None

    measurements: dict | None = None
    if measurements_json:
        try:
            measurements = MeasurementsIn(**json.loads(measurements_json)).model_dump(
                exclude_none=True
            )
        except Exception as exc:
            raise HTTPException(400, detail=f"Invalid measurements JSON: {exc}") from exc

    t0 = time.perf_counter()
    try:
        result = predictor.predict(
            lateral_path=lat_path,
            top_path=top_path,
            back_path=bak_path,
            measurements=measurements,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, detail=f"Inference failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    patient_id = None
    if patient_code:
        patient_id = PatientRepository(db).upsert_by_code(patient_code).id

    used = result.measurements_used
    try:
        row = ClassificationRepository(db).create(
            patient_id=patient_id,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            class_probs=result.class_probabilities,
            severity_band=result.severity_band,
            rule_based_label=result.predicted_class,
            notes="\n".join(result.notes) if result.notes else None,
            lateral_image_path=str(lat_path) if lat_path else None,
            top_image_path=str(top_path) if top_path else None,
            back_image_path=str(bak_path) if bak_path else None,
            calcaneal_inclination_deg=used.get("calcaneal_inclination_deg"),
            heel_angle_deg=used.get("heel_angle_deg"),
            arch_height_cm=used.get("arch_height_cm"),
            kite_angle_deg=used.get("kite_angle_deg"),
            first_metatarsal_talus_deg=used.get("first_metatarsal_talus_deg"),
            measurements_were_provided=(result.classification_source == "measured"),
            arch_support_height=result.insole_configuration.get("arch_support_height"),
            heel_cup_depth=result.insole_configuration.get("heel_cup_depth"),
            medial_post_strength=result.insole_configuration.get("medial_post_strength"),
            lateral_wedge_strength=result.insole_configuration.get("lateral_wedge_strength"),
            forefoot_cushioning=result.insole_configuration.get("forefoot_cushioning"),
            inference_time_ms=elapsed_ms,
            model_version="v0.2.0-measurement-first",
        )
        record_id = row.id
    except Exception as exc:
        logger.warning("Could not persist classification: {}", exc)
        record_id = None

    return ClassificationOut(
        id=record_id,
        predicted_class=result.predicted_class,
        predicted_class_idx=result.predicted_class_idx,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        severity_band=result.severity_band,
        rule_based_label=result.predicted_class,
        measurements_predicted=result.measurements_used,
        measurements_provided=result.measurements_provided,
        insole_configuration=InsoleConfigOut(**result.insole_configuration),
        notes=result.notes,
        inference_time_ms=elapsed_ms,
        model_version="v0.2.0-measurement-first",
        classification_source=result.classification_source,
        measurements_estimated=result.measurements_estimated,
    )
