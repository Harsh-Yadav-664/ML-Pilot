"""Run Alembic migrations for the metadata DB (on app start-up, in tests and from the CLI).

Alembic owns the schema; the app never creates tables from the ORM metadata directly.
A database made before migrations existed has the initial tables but no
``alembic_version``: it is stamped at the initial revision and then upgraded,
so existing rows are kept.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"
INITIAL_REVISION = "0001"


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", database_url or settings.DATABASE_URL)
    cfg.attributes["skip_logging_config"] = True  # keep the app's logging setup
    return cfg


def _table_names(database_url: str) -> set[str]:
    async def read() -> set[str]:
        engine = create_async_engine(database_url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return set(await conn.run_sync(lambda c: inspect(c).get_table_names()))
        finally:
            await engine.dispose()

    return asyncio.run(read())


def upgrade_to_head(database_url: str | None = None) -> None:
    """Bring the metadata DB to the latest schema. Blocking: call it via asyncio.to_thread."""
    url = database_url or settings.DATABASE_URL
    cfg = alembic_config(url)
    tables = _table_names(url)
    if "experiments" in tables and "alembic_version" not in tables:
        logger.info("Existing database without migration history: stamping %s", INITIAL_REVISION)
        command.stamp(cfg, INITIAL_REVISION)
    command.upgrade(cfg, "head")
