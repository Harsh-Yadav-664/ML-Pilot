"""An invalid LLM formula rejects the experiment with its reason; it never becomes a zero column."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy import select

from app.db.models.experiment import Experiment
from app.db.session import get_db
from app.main import app
from ml.agents.decision_agent import DecisionAgent
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.planner import ExperimentPlanner
from ml.experiments.schema import ExperimentSpec
from tests.fixtures.gateway import stub_gateway
from tests.integration.test_agent_loop_telecom import session_factory  # noqa: F401  (fixture)

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


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
    session_factory,  # noqa: F811 - pytest fixture imported above
    monkeypatch,
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
    result = await DecisionAgent(stub_gateway(), settings=None).run_optimization_loop(
        str(SAMPLE), "Churn", n_hypotheses=1
    )

    async with session_factory() as session:
        exps = (await session.execute(select(Experiment))).scalars().all()
    bad = next(e for e in exps if e.parent_id)
    assert bad.status == "rejected_invalid" and bad.decision == "reject"
    assert (
        "formula is invalid" in bad.decision_reason
        and "Attribute is not allowed" in bad.decision_reason
    )
    assert not bad.metrics
    assert result["champion_id"] == next(e.id for e in exps if not e.parent_id)

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tree = (
                await client.get(
                    "/api/v1/ui/experiments/tree", params={"dataset_path": str(SAMPLE)}
                )
            ).json()
    finally:
        app.dependency_overrides.clear()
    node = next(e for e in tree if e["id"] == bad.id)
    assert node["status"] == "rejected_invalid" and "Attribute is not allowed" in node["error"]
