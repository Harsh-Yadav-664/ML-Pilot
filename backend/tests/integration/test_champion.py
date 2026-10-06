"""Accepted features accumulate: the champion is the model that was actually measured."""

from __future__ import annotations

import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import ml.experiments.executor as executor_module
from app.core import datasets
from app.db.models.experiment import Experiment
from ml.agents.decision_agent import DecisionAgent
from ml.experiments.acceptance import GainResult
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.api import API, load_sample
from tests.fixtures.gateway import stub_gateway

FEATURES = [
    {"name": "charge_x_tenure", "formula": "MonthlyCharges * tenure"},
    {"name": "avg_charge", "formula": "TotalCharges / (tenure + 1)"},
    {"name": "charge_gap", "formula": "MonthlyCharges - TotalCharges / (tenure + 1)"},
]


def _always_accept(X_base, X_candidate, y, make_pipeline, rule=None, random_state=42):
    return GainResult(
        "roc_auc", [0.8], [0.9], 0.1, 0.0, (0.1, 0.1), 0.002, True, {"forced": "test"}
    )


async def test_three_accepted_features_accumulate_in_the_champion(
    client, project_id, metadata_db_url, monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "optuna", None)
    # This test is about the champion mechanics, so acceptance is forced; the rule itself
    # is tested in tests/unit/test_acceptance.py.
    monkeypatch.setattr(executor_module, "compare_feature_sets", _always_accept)
    proposals = iter(FEATURES)

    async def propose(self, **kwargs):
        f = next(proposals)
        return {**f, "reason": f"try {f['name']}", "non_redundant_reasoning": "new"}

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)

    version = (await load_sample(client, project_id))["data_version_id"]
    path = datasets.VERSIONS_DIR / f"{version}.csv"
    result = await DecisionAgent(stub_gateway(), settings=None).run_optimization_loop(
        str(path), "Churn", n_hypotheses=3, project_id=project_id
    )

    assert [f["name"] for f in result["champion_features"]] == [f["name"] for f in FEATURES]
    factory = async_sessionmaker(create_async_engine(metadata_db_url, poolclass=NullPool))
    async with factory() as session:
        rows = await session.execute(select(Experiment).where(Experiment.project_id == project_id))
        exps = {e.id: e for e in rows.scalars().all()}
    champion = exps[result["champion_id"]]
    # The champion run itself contained all three features...
    used = [f["name"] for f in champion.parameters["features"]] + [
        champion.parameters["feature_name"]
    ]
    assert used == [f["name"] for f in FEATURES]
    for f in FEATURES:
        assert f["name"] in champion.parameters["feature_columns"]
    # ...and its metrics are that run's metrics.
    assert champion.status == "completed" and "test_f1" in champion.metrics
    # Lineage: baseline -> 1 -> 2 -> 3.
    chain, cur = [], champion
    while cur is not None:
        chain.append(cur)
        cur = exps.get(cur.parent_id) if cur.parent_id else None
    assert len(chain) == 4 and chain[-1].parent_id is None

    base = f"{API}/projects/{project_id}/experiments"
    tree = (await client.get(base, params={"data_version_id": version})).json()
    exported = await client.get(f"{base}/champion/export", params={"data_version_id": version})
    flagged = [e["id"] for e in tree if e["champion"]]
    assert flagged == [champion.id]
    assert sum(e["on_champion_path"] for e in tree) == 4
    assert exported.status_code == 200, exported.text
    script = exported.json()["script"]
    assert champion.id in script and f'model_name = "{champion.model_name}"' in script
    for f in FEATURES:
        assert f["name"] in script
    # The exported champion script rebuilds all three features and trains.
    import subprocess

    path = tmp_path / "champion.py"
    path.write_text(script)
    # Blocking is fine here: the test waits for the exported script on purpose.
    run = subprocess.run(  # noqa: ASYNC221
        [sys.executable, str(path)], capture_output=True, text=True, timeout=300, check=False
    )
    assert run.returncode == 0, run.stderr[-2000:]
