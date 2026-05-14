"""
FastAPI application — the HTTP boundary of the system.

Layout
------
    /api/health           -> liveness + model/db status
    /api/classify         -> multipart upload of 3 images + measurements
    /api/training/runs    -> list/start/status of training runs
    /api/data/summary     -> quick stats over the data folder
    /api/patients         -> CRUD for patient records

The FastAPI app is run by `uvicorn backend.server.main:app` and also serves
as the bridge between the PySide6 desktop GUI and the backend AI model.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.database.connection import create_all, get_engine, is_connected
from backend.server.middleware import LoggingMiddleware, RequestIDMiddleware
from backend.server.routes import classification, data_router, health, patients, training


# ---------------------------------------------------------------------------
# App state container (lives on app.state)
# ---------------------------------------------------------------------------
class AppState:
    predictor = None      # backend.model.Predictor
    model_loaded: bool = False
    device: str = "cpu"
    db_connected: bool = False


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.app = state

    # ----- DB -----
    try:
        if os.getenv("AUTO_CREATE_TABLES", "1") == "1":
            create_all()
            logger.info("Database tables ensured.")
        state.db_connected = is_connected()
    except Exception as exc:
        logger.warning("Database unavailable: {}", exc)
        state.db_connected = False

    # ----- Predictor (lazy: only if checkpoint exists) -----
    try:
        from backend.model.config import InferenceConfig
        from backend.model.inference.predictor import Predictor

        ckpt = Path(os.getenv("DEFAULT_CHECKPOINT_PATH", "backend/model/checkpoints/best.pt"))
        cfg = InferenceConfig(checkpoint_path=ckpt)
        state.predictor = Predictor(cfg)
        state.model_loaded = ckpt.exists()
        state.device = str(state.predictor.device)
        logger.info(
            "Predictor ready on {} ({}).",
            state.device,
            "trained weights" if state.model_loaded else "RANDOM weights — train first!",
        )
    except Exception as exc:
        logger.warning("Predictor init failed: {}", exc)
        state.predictor = None
        state.model_loaded = False

    yield

    # ----- Shutdown -----
    try:
        get_engine().dispose()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Insole Foot Classification API",
        version="0.1.0",
        description="AI-powered foot-classification service for insole recommendation.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware order matters: outer to inner.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(classification.router, prefix="/api", tags=["classification"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(data_router.router, prefix="/api/data", tags=["data"])
    app.include_router(patients.router, prefix="/api/patients", tags=["patients"])

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "insole-foot-classification",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


# ASGI entrypoint used by uvicorn.
app = create_app()


# ---------------------------------------------------------------------------
# `python -m backend.server.main` runs uvicorn directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
