"""Database engine, session, and dependency factory."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base


# ---------------------------------------------------------------------------
# Engine (singleton)
# ---------------------------------------------------------------------------
def _build_dsn() -> str:
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    user = os.getenv("POSTGRES_USER", "insole_admin")
    pw = os.getenv("POSTGRES_PASSWORD", "change_me_securely")
    db = os.getenv("POSTGRES_DB", "insole_db")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Lazy-create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        dsn = _build_dsn()
        logger.info("Connecting to database (host={})", dsn.split("@")[-1].split("/")[0])
        _engine = create_engine(
            dsn,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            future=True,
        )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, future=True
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency."""
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Standalone context manager (used outside FastAPI, e.g. seeding)."""
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_all() -> None:
    """Create all tables — convenience for local dev without Alembic."""
    Base.metadata.create_all(bind=get_engine())


def is_connected() -> bool:
    """Quick liveness probe. SQLAlchemy 2.0 requires text() for raw SQL."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False