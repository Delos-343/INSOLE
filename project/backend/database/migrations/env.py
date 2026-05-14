"""Alembic environment — uses our SQLAlchemy metadata."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.database.connection import _build_dsn
from backend.database.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject our DSN into the alembic config so users don't need to duplicate it.
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL") or _build_dsn())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
