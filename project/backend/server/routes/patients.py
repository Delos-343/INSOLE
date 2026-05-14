"""Patient CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.repositories.patient_repo import PatientRepository
from backend.database.schemas import PatientIn, PatientOut

router = APIRouter()


@router.post("", response_model=PatientOut, status_code=201)
async def create_patient(p: PatientIn, db: Session = Depends(get_db)) -> PatientOut:
    row = PatientRepository(db).upsert_by_code(**p.model_dump())
    return PatientOut.model_validate(row)


@router.get("", response_model=list[PatientOut])
async def list_patients(limit: int = 100, db: Session = Depends(get_db)):
    return [PatientOut.model_validate(r) for r in PatientRepository(db).list_all(limit)]


@router.get("/{code}", response_model=PatientOut)
async def get_patient(code: str, db: Session = Depends(get_db)) -> PatientOut:
    row = PatientRepository(db).get_by_code(code)
    if row is None:
        raise HTTPException(404, detail="Patient not found.")
    return PatientOut.model_validate(row)
