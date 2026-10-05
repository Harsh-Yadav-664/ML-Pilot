"""API tests for the chat endpoints and Auto-Clean, using the offline stub provider."""
from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.core.datasets as datasets
from app.api.deps import get_gateway
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from tests.fixtures.gateway import stub_gateway


@pytest.fixture
async def session_factory(tmp_path):
    import app.db.models  # noqa: F401  (register tables)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


@pytest.fixture
async def client(session_factory) -> AsyncGenerator[httpx.AsyncClient]:
    async def override_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_gateway] = stub_gateway
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_chat_ask_answers_with_stub(client):
    resp = await client.post("/api/v1/chat/ask", json={"query": "Which model was best?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert "grounding_context" in body


async def test_chat_settings_returns_structured_settings(client):
    resp = await client.post("/api/v1/chat/settings", json={"query": "only use tree models"})
    assert resp.status_code == 200, resp.text
    parsed = resp.json()["parsed_settings"]
    assert set(parsed) == {"model_family_restriction", "optimization_metric", "max_runtime_minutes"}


async def test_chat_debrief_for_existing_experiment(client, session_factory):
    async with session_factory() as session:
        exp = await ExperimentService(session).create(
            ExperimentCreate(
                project_id="demo-project-id",
                dataset_version="data.csv",
                hypothesis="baseline",
                change_description="baseline",
                model_name="RandomForestClassifier",
            )
        )
        await session.commit()

    resp = await client.get(f"/api/v1/chat/debrief/{exp.id}")
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json()["debrief"], str) and resp.json()["debrief"]


async def test_chat_debrief_unknown_experiment_is_404(client):
    resp = await client.get("/api/v1/chat/debrief/does-not-exist")
    assert resp.status_code == 404


async def test_auto_clean_surfaces_llm_failure(client, tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "ALLOWED_DATA_DIRS", [tmp_path])
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"age": [20, 30, 40, 50], "target": [0, 1, 0, 1]}).to_csv(csv_path, index=False)

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    gateway = stub_gateway()
    monkeypatch.setattr(gateway.providers["stub"], "complete_structured", boom)
    app.dependency_overrides[get_gateway] = lambda: gateway

    resp = await client.post(
        "/api/v1/ui/agent/auto-clean",
        json={"dataset_path": str(csv_path), "target_column": "target"},
    )
    assert resp.status_code == 502
    assert "AI cleaning strategy failed" in resp.json()["detail"]
