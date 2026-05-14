"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_predictor(request: Request):
    """Return the global Predictor or 503 if it failed to load."""
    state = getattr(request.app.state, "app", None)
    if state is None or state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised. Train a model first.",
        )
    return state.predictor


def get_app_state(request: Request):
    """Return the AppState container."""
    return request.app.state.app
