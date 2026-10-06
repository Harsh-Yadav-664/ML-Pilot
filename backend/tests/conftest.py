"""Shared test isolation: no test writes to the real metadata DB or data/versions."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import app.db.session as db_session
import ml.agents.decision_agent as decision_agent_module
from app.core import datasets
from app.db.migrations import upgrade_to_head
from app.db.models import Job
from app.db.session import get_db
from app.jobs import handlers  # noqa: F401  (registers the job kinds)
from app.jobs.runner import JobWorker
from app.main import app
from tests.fixtures.api import create_project


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
    """Point data versions at a temp dir, and API and background sessions at the temp DB.

    Tests that need their own DB still override these themselves; that wins.
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
    # Background runs and the agent open their own sessions.
    monkeypatch.setattr(db_session, "AsyncSessionLocal", factory)
    monkeypatch.setattr(decision_agent_module, "AsyncSessionLocal", factory)
    yield versions
    if app.dependency_overrides.get(get_db) is test_db:
        del app.dependency_overrides[get_db]


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def project_id(client: httpx.AsyncClient) -> str:
    """A fresh project, created through the API like the UI does."""
    return await create_project(client)


@pytest.fixture(autouse=True)
async def job_worker(isolated_storage: Path, metadata_db_url: str) -> AsyncGenerator[JobWorker]:
    """The job runner, as the app starts it, against the test DB (short heartbeat)."""
    worker = JobWorker(concurrency=2, heartbeat=0.2, stale_after=30, idle_poll=0.1)
    await worker.start()
    yield worker
    await worker.stop()
    # Jobs a test queued but didn't wait for must not run during the next test.
    engine = create_async_engine(metadata_db_url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(update(Job).where(Job.status == "queued").values(status="cancelled"))
    await engine.dispose()
