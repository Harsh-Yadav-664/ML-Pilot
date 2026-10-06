"""The debrief only talks about real columns of the dataset (no fake SHAP)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_gateway
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from tests.fixtures.gateway import stub_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
async def session_factory(tmp_path):
    import app.db.models as _models  # noqa: F401  (register tables)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def test_debrief_mentions_only_real_columns(session_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)  # keep the run fast
    df = pd.read_csv(SAMPLE)
    spec = ExperimentSpec(
        id="exp_debrief",
        project_id="demo-project-id",
        dataset_version=str(SAMPLE),
        hypothesis="baseline",
        change_description="baseline",
        model_name="RandomForestClassifier",
        parameters={
            "target_column": "Churn",
            "model_params": {"n_estimators": 20},
            "ensemble": False,
        },
    )
    result = await LocalExperimentExecutor(lambda _: df.drop(columns=["customerID"])).run(spec)
    importances = result.parameters["feature_importances"]
    assert importances, result.parameters["importance_method"]
    assert set(importances) <= set(df.columns) - {"Churn"}
    assert abs(sum(importances.values()) - 1) < 0.05 or len(importances) == 10

    async with session_factory() as session:
        exp = await ExperimentService(session).create(
            ExperimentCreate(
                project_id="demo-project-id",
                dataset_version=str(SAMPLE),
                hypothesis="baseline",
                change_description="baseline",
                model_name="RandomForestClassifier",
                parameters=result.parameters,
            )
        )
        exp.metrics = result.metrics
        await session.commit()

    gateway = stub_gateway()
    prompts: list[str] = []
    real_complete = gateway.complete

    async def capture(task_type, prompt, **kwargs):
        prompts.append(prompt)
        return await real_complete(task_type, prompt, **kwargs)

    gateway.complete = capture

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_gateway] = lambda: gateway
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/v1/chat/debrief/{exp.id}")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "shap_values" not in body
    assert set(body["feature_importances"]) <= set(df.columns)
    assert "not SHAP" in body["importance_method"]
    # The prompt names the real top columns and none of the old fake ones.
    assert next(iter(importances)) in prompts[0]
    for fake in ("'age'", "'balance'", "'is_active'"):
        assert fake not in prompts[0]


async def test_debrief_without_importances_says_not_available(session_factory):
    async with session_factory() as session:
        exp = await ExperimentService(session).create(
            ExperimentCreate(
                project_id="demo-project-id",
                dataset_version="x.csv",
                hypothesis="h",
                change_description="c",
                model_name="RandomForestClassifier",
            )
        )
        await session.commit()

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_gateway] = stub_gateway
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/v1/chat/debrief/{exp.id}")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["feature_importances"] == {}
    assert resp.json()["importance_method"].startswith("not available")
