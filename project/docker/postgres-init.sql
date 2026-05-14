-- ============================================================================
-- Initial Postgres setup — runs once when the container is first created.
-- Most schema work happens via Alembic; this file only handles things that
-- need superuser privileges (extensions, default settings).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sane timezone.
SET timezone = 'UTC';

-- Helpful: log slow queries when running in containerised dev.
ALTER SYSTEM SET log_min_duration_statement = 500;
