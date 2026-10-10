"""Accepted features accumulate: the champion is the model that was actually measured."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from ml.agents import run_loop
from ml.experiments.acceptance import DEFAULT_RULE
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.api import API, load_sample
from tests.integration.test_agent_loop_telecom import follow_job

FEATURES = [
    {"name": "charge_x_tenure", "formula": "MonthlyCharges * tenure"},
    {"name": "avg_charge", "formula": "TotalCharges / (tenure + 1)"},
    {"name": "charge_gap", "formula": "MonthlyCharges - TotalCharges / (tenure + 1)"},
]


async def test_three_accepted_features_accumulate_in_the_champion(
    client, project_id, monkeypatch, tmp_path
):
    # This test is about the champion mechanics, so any gain is accepted (margin below zero);
    # the rule itself is tested in tests/unit/test_acceptance.py and tests/unit/test_feature_gain.py.
    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    proposals = iter(FEATURES)

    async def propose(self, **kwargs):
        f = next(proposals)
        return {**f, "reason": f"try {f['name']}", "non_redundant_reasoning": "new"}

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)
    calls: list[int] = []
    real = run_loop.score_test

    def spy(model, features: pd.DataFrame, y: np.ndarray, **kwargs):
        calls.append(len(y))
        return real(model, features, y, **kwargs)

    monkeypatch.setattr(run_loop, "score_test", spy)

    base = f"{API}/projects/{project_id}/experiments"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{API}/projects/{project_id}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 3},
    )
    status, _ = await follow_job(client, f"{API}/projects/{project_id}", job.json()["id"])
    assert status["status"] == "succeeded", status
    result = status["result"]

    assert [f["name"] for f in result["champion_features"]] == [f["name"] for f in FEATURES]
    assert result["winner_features"] == [f["name"] for f in FEATURES]
    tree = (await client.get(base, params={"data_version_id": version})).json()
    exps = {n["id"]: (await client.get(f"{base}/{n['id']}")).json() for n in tree}
    champion = exps[result["champion_id"]]
    # The champion run itself contained all three features...
    used = [f["name"] for f in champion["parameters"]["features"]] + [
        champion["parameters"]["feature_name"]
    ]
    assert used == [f["name"] for f in FEATURES]
    for f in FEATURES:
        assert f["name"] in champion["parameters"]["feature_columns"]
    # ...and its metrics are that run's metrics: the test rows were scored once, for it only.
    assert champion["status"] == "completed" and "test_f1" in champion["metrics"]
    assert len(calls) == 1
    assert [e["id"] for e in exps.values() if any(k.startswith("test_") for k in e["metrics"])] == [
        champion["id"]
    ]
    # Lineage: baseline -> 1 -> 2 -> 3.
    chain, cur = [], champion
    while cur is not None:
        chain.append(cur)
        cur = exps.get(cur["parent_id"]) if cur["parent_id"] else None
    assert len(chain) == 4 and chain[-1]["parent_id"] is None
    assert [e["decision"] for e in reversed(chain[:-1])] == ["keep"] * 3

    exported = await client.get(f"{base}/champion/export", params={"data_version_id": version})
    flagged = [e["id"] for e in tree if e["champion"]]
    assert flagged == [champion["id"]]
    assert sum(e["on_champion_path"] for e in tree) == 4
    assert exported.status_code == 200, exported.text
    script = exported.json()["script"]
    assert champion["id"] in script and f'model_name = "{champion["model_name"]}"' in script
    for f in FEATURES:
        assert f["name"] in script
    # The exported champion script rebuilds all three features and trains.
    path = tmp_path / "champion.py"
    path.write_text(script)
    # Blocking is fine here: the test waits for the exported script on purpose.
    run = subprocess.run(  # noqa: ASYNC221
        [sys.executable, str(path)], capture_output=True, text=True, timeout=300, check=False
    )
    assert run.returncode == 0, run.stderr[-2000:]
