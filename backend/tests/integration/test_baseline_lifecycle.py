"""POST /ui/experiments/baseline must reach a final status, never stay 'created'."""
from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.session as db_session
from app.db.base import Base
from app.db.session import get_db
from app.main import app

FINAL = {"completed", "failed"}


@pytest.fixture
async def client(tmp_path, monkeypatch):
    import app.db.models as _models  # noqa: F401  (register tables)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    # Background runs open their own sessions through app.db.session.AsyncSessionLocal.
    monkeypatch.setattr(db_session, "AsyncSessionLocal", factory)

    async def override_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def small_csv(tmp_path):
    path = tmp_path / "small.csv"
    rows = 60
    pd.DataFrame({
        "age": list(range(rows)),
        "plan": ["a", "b", "c"] * (rows // 3),
        "churn": ["Yes" if i % 3 == 0 else "No" for i in range(rows)],
    }).to_csv(path, index=False)
    return path


async def wait_for_final_status(client: httpx.AsyncClient, exp_id: str, timeout: float = 60) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        resp = await client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in FINAL or loop.time() > deadline:
            return body
        await asyncio.sleep(0.05)


async def test_baseline_reaches_completed_20_times(client, small_csv):
    for _ in range(20):
        resp = await client.post(
            "/api/v1/ui/experiments/baseline",
            json={"dataset_path": str(small_csv), "target_column": "churn"},
        )
        assert resp.status_code == 200, resp.text
        body = await wait_for_final_status(client, resp.json()["id"])
        assert body["status"] == "completed", body.get("decision_reason")


async def test_baseline_failure_is_recorded_with_message(client, tmp_path):
    resp = await client.post(
        "/api/v1/ui/experiments/baseline",
        json={"dataset_path": str(tmp_path / "missing.csv"), "target_column": "churn"},
    )
    assert resp.status_code == 200, resp.text
    body = await wait_for_final_status(client, resp.json()["id"])
    assert body["status"] == "failed"
    assert body["decision_reason"]
