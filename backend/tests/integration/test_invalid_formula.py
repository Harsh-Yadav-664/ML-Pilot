"""An invalid LLM formula rejects the experiment with its reason; it never becomes a zero column."""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core import datasets
from app.db.models.experiment import Experiment
from ml.agents.decision_agent import DecisionAgent
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.planner import ExperimentPlanner
from ml.experiments.schema import ExperimentSpec
from tests.fixtures.api import API, load_sample
from tests.fixtures.gateway import stub_gateway


async def test_executor_rejects_invalid_formula_without_training(monkeypatch):
    fits = []
    monkeypatch.setattr(Pipeline, "fit", lambda self, *a, **k: fits.append(1))
    df = pd.DataFrame({"x": range(50), "target": [0, 1] * 25})
    spec = ExperimentSpec(
        id="bad",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name="LogisticRegression",
        parameters={
            "target_column": "target",
            "feature_name": "evil",
            "formula": "__import__('os').getcwd()",
        },
    )
    result = await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)
    assert result.status.value == "rejected_invalid"
    assert result.decision.value == "reject"
    assert result.metrics == {} and fits == []
    assert "not allowed" in result.parameters["invalid_formula"]["reason"]
    assert "Invalid formula for 'evil'" in result.decision_reason


async def test_agent_records_invalid_formula_and_keeps_baseline_champion(
    client, project_id, metadata_db_url, monkeypatch
):
    monkeypatch.setitem(sys.modules, "optuna", None)

    async def propose(self, **kwargs):
        return {
            "name": "evil",
            "formula": "().__class__",
            "reason": "r",
            "non_redundant_reasoning": "n",
        }

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)
    version = (await load_sample(client, project_id))["data_version_id"]
    path = datasets.VERSIONS_DIR / f"{version}.csv"
    result = await DecisionAgent(stub_gateway(), settings=None).run_optimization_loop(
        str(path), "Churn", n_hypotheses=1, project_id=project_id
    )

    factory = async_sessionmaker(create_async_engine(metadata_db_url, poolclass=NullPool))
    async with factory() as session:
        rows = await session.execute(select(Experiment).where(Experiment.project_id == project_id))
        exps = rows.scalars().all()
    bad = next(e for e in exps if e.parent_id)
    assert bad.status == "rejected_invalid" and bad.decision == "reject"
    assert (
        "formula is invalid" in bad.decision_reason
        and "Attribute is not allowed" in bad.decision_reason
    )
    assert not bad.metrics
    assert result["champion_id"] == next(e.id for e in exps if not e.parent_id)

    tree = (
        await client.get(
            f"{API}/projects/{project_id}/experiments", params={"data_version_id": version}
        )
    ).json()
    node = next(e for e in tree if e["id"] == bad.id)
    assert node["status"] == "rejected_invalid" and "Attribute is not allowed" in node["error"]
