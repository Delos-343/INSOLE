"""Health/liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.database.schemas import HealthOut
from backend.server.dependencies import get_app_state

router = APIRouter()


@router.get("/health", response_model=HealthOut)
async def health(state=Depends(get_app_state)) -> HealthOut:
    return HealthOut(
        model_loaded=bool(state and state.model_loaded),
        device=str(state.device if state else "cpu"),
        db_connected=bool(state and state.db_connected),
    )
