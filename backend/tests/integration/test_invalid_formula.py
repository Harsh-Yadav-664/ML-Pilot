"""An invalid LLM formula rejects the experiment with its reason; it never becomes a zero column."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.planner import ExperimentPlanner
from ml.experiments.schema import ExperimentSpec
from tests.fixtures.api import API, load_sample
from tests.integration.test_agent_loop_telecom import follow_job


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
    client, project_id, monkeypatch
):
    async def propose(self, **kwargs):
        return {
            "name": "evil",
            "formula": "().__class__",
            "reason": "r",
            "non_redundant_reasoning": "n",
        }

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)
    base = f"{API}/projects/{project_id}"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 1},
    )
    status, _ = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "succeeded", status
    result = status["result"]

    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    exps = [(await client.get(f"{base}/experiments/{n['id']}")).json() for n in tree]
    bad = next(e for e in exps if e["parent_id"])
    assert bad["status"] == "rejected_invalid" and bad["decision"] == "reject"
    assert (
        "formula is invalid" in bad["decision_reason"]
        and "Attribute is not allowed" in bad["decision_reason"]
    )
    assert not bad["metrics"]  # nothing was trained
    assert bad["parameters"]["invalid_formula"]["formula"] == "().__class__"
    baseline = next(e for e in exps if not e["parent_id"])
    assert result["champion_id"] == baseline["id"]
    assert result["winner_features"] == []
    node = next(e for e in tree if e["id"] == bad["id"])
    assert node["status"] == "rejected_invalid" and "Attribute is not allowed" in node["error"]
    # the champion is still the baseline, and it is the row that holds the test metrics
    champion = next(e for e in tree if e["champion"])
    assert champion["id"] == baseline["id"]
    assert baseline["metrics"]["test_pr_auc"] > baseline["metrics"]["test_base_rate"]


async def test_a_planner_that_fails_is_recorded_and_the_run_goes_on(
    client, project_id, monkeypatch
):
    calls = []

    async def propose(self, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("all providers failed")
        return {"formula": "MonthlyCharges * tenure"}  # no name: not a usable answer

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)
    base = f"{API}/projects/{project_id}"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 2},
    )
    status, events = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "succeeded", status
    infos = status["result"]["experiments"]
    assert [i["decision"] for i in infos] == ["reject", "reject"]
    assert "all providers failed" in infos[0]["reason"]
    assert infos[0]["hypothesis_llm"]["decision_mode"] == "fallback"
    assert [e["type"] for e in events].count("decision") == 2
    # nothing was tested: the experiment tree is the baseline alone, and it is the champion
    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    assert len(tree) == 1 and tree[0]["champion"]


async def test_a_target_with_three_classes_fails_the_job_with_the_reason(client, project_id):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    df["kind"] = np.where(df["a"] > 0.5, "x", np.where(df["b"] > 0, "y", "z"))
    up = await client.post(
        f"{API}/projects/{project_id}/datasets/upload",
        files={"file": ("d.csv", df.to_csv(index=False).encode())},
    )
    base = f"{API}/projects/{project_id}"
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": up.json()["data_version_id"], "target_column": "kind"},
    )
    status, _ = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "failed" and "two-class" in status["error"], status
