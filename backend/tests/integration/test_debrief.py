"""The debrief only talks about real columns of the dataset (no fake SHAP)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ai.gateway import AIGateway
from app.api.deps import get_gateway
from app.main import app
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from tests.fixtures.api import API
from tests.fixtures.gateway import stub_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
def session_factory(metadata_db_url):
    return async_sessionmaker(
        create_async_engine(metadata_db_url, poolclass=NullPool), expire_on_commit=False
    )


@pytest.fixture(autouse=True)
def gateway() -> Iterator[AIGateway]:
    gw = stub_gateway()
    app.dependency_overrides[get_gateway] = lambda: gw
    yield gw
    app.dependency_overrides.pop(get_gateway, None)


async def test_debrief_mentions_only_real_columns(
    client, project_id, session_factory, gateway, monkeypatch
):
    monkeypatch.setitem(sys.modules, "optuna", None)  # keep the run fast
    df = pd.read_csv(SAMPLE)
    spec = ExperimentSpec(
        id="exp_debrief",
        project_id="p",
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
                project_id=project_id,
                dataset_version=str(SAMPLE),
                hypothesis="baseline",
                change_description="baseline",
                model_name="RandomForestClassifier",
                parameters=result.parameters,
            )
        )
        exp.metrics = result.metrics
        await session.commit()

    prompts: list[str] = []
    real_complete = gateway.complete

    async def capture(task_type, prompt, **kwargs):
        prompts.append(prompt.text)
        return await real_complete(task_type, prompt, **kwargs)

    monkeypatch.setattr(gateway, "complete", capture)
    resp = await client.get(f"{API}/projects/{project_id}/experiments/{exp.id}/debrief")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "shap_values" not in body
    assert set(body["feature_importances"]) <= set(df.columns)
    assert "not SHAP" in body["importance_method"]
    # The prompt names the real top columns and none of the old fake ones.
    assert next(iter(importances)) in prompts[0]
    for fake in ('"age"', '"balance"', '"is_active"'):
        assert fake not in prompts[0]


async def test_debrief_without_importances_says_not_available(client, project_id, session_factory):
    async with session_factory() as session:
        exp = await ExperimentService(session).create(
            ExperimentCreate(
                project_id=project_id,
                dataset_version="x.csv",
                hypothesis="h",
                change_description="c",
                model_name="RandomForestClassifier",
            )
        )
        await session.commit()

    resp = await client.get(f"{API}/projects/{project_id}/experiments/{exp.id}/debrief")
    assert resp.status_code == 200
    assert resp.json()["feature_importances"] == {}
    assert resp.json()["importance_method"].startswith("not available")
