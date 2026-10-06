"""Shared test isolation: no test writes to the real metadata DB or data/versions."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core import datasets
from app.db.migrations import upgrade_to_head
from app.db.session import get_db
from app.main import app


@pytest.fixture(scope="session")
def metadata_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A temporary metadata DB built by the real migrations."""
    url = f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('metadata') / 'mlpilot.db'}"
    upgrade_to_head(url)
    return url


@pytest.fixture(autouse=True)
def isolated_storage(
    metadata_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point data versions at a temp dir and API sessions at the temporary DB.

    Tests that need their own DB still override get_db themselves; that wins.
    """
    versions = tmp_path / "versions"
    monkeypatch.setattr(datasets, "VERSIONS_DIR", versions)
    factory = async_sessionmaker(
        create_async_engine(metadata_db_url, poolclass=NullPool), expire_on_commit=False
    )

    async def test_db() -> AsyncGenerator[AsyncSession]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = test_db
    yield versions
    if app.dependency_overrides.get(get_db) is test_db:
        del app.dependency_overrides[get_db]
