"""The single-table agent loop on the bundled telecom sample (string Yes/No target), end to end.

Since #151 this runs on the one run loop (``ml/agents/run_loop.py``): the same acceptance rule
and the same single scoring of the test rows as a relational run.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from app.jobs import handlers
from ml.agents import run_loop
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.api import API, load_sample
from tests.fixtures.gateway import stub_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
def test_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many rows each scoring of the test rows was given: the spy."""
    calls: list[int] = []
    real = run_loop.score_test

    def spy(model: Any, features: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        calls.append(len(y))
        return real(model, features, y, **kwargs)

    monkeypatch.setattr(run_loop, "score_test", spy)
    return calls


@pytest.fixture
def real_formulas(monkeypatch):
    """The offline stub has no planner output, so propose real formulas on this sample.
    (Without this the planner reports the offline fallback and the loop proposes nothing.)"""
    proposals = iter(
        [
            {"name": "charge_x_tenure", "formula": "MonthlyCharges * tenure"},
            {"name": "avg_charge", "formula": "TotalCharges / (tenure + 1)"},
        ]
    )

    async def propose(self, **kwargs):
        return {**next(proposals), "reason": "test proposal", "non_redundant_reasoning": "new"}

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)


async def test_baseline_and_two_iterations_complete_and_export_decodes(
    client, project_id, tmp_path, real_formulas, test_calls
):
    """The Phase 0 end-to-end flow, driven only through the project API like the UI."""
    base = f"{API}/projects/{project_id}"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 2},
    )
    assert job.status_code == 200, job.text
    assert job.json()["status"] == "queued"
    status, events = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "succeeded", status
    assert status["progress"] == 1.0
    result = status["result"]

    # Progress came through the ordered event log: seq 1, 2, 3, ... with no gaps.
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    types = [e["type"] for e in events]
    assert types[0] == "step" and types.count("proposal") == 2 and types.count("decision") == 2
    assert types.index("cv_result") < types.index("proposal") < types.index("decision")
    # the events of the one loop: a decision per round, and the end of the run
    assert types.count("feature_decision") == 2 and types[-1] == "run_finished"
    steps = [
        e["payload"]["name"] for e in events if e["type"] == "step" and "progress" in e["payload"]
    ]
    assert [s for s in steps if s.startswith("Round")] == [
        "Round 1: asking for a feature",
        "Round 2: asking for a feature",
    ]

    # The test rows were scored exactly once for the whole run, by the loop's score_test.
    assert len(test_calls) == 1
    assert result["test"]["n_test"] == test_calls[0]
    assert result["status"] == "completed" and result["stop_reason"] == "max_rounds"
    print(
        f"\n{result['summary']} Test PR-AUC {result['test']['pr_auc']:.4f} "
        f"(base rate {result['test']['base_rate']:.4f}); score_test called {len(test_calls)} "
        f"time(s) on {test_calls[0]} rows; "
        + ", ".join(f"{i['feature_name']}: {i['decision']}" for i in result["experiments"])
    )

    assert len(result["experiments"]) == 2
    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    assert len(tree) == 3  # baseline + 2 iterations
    exps = [(await client.get(f"{base}/experiments/{n['id']}")).json() for n in tree]
    by_id = {e["id"]: e for e in exps}
    champion = by_id[result["champion_id"]]
    for exp in exps:
        assert exp["status"] == "completed", exp["decision_reason"]
        # Default positive class is the minority class ("Yes" in this sample)
        assert exp["parameters"]["target_encoding"] == {
            "classes": ["No", "Yes"],
            "positive_class": "Yes",
        }
        assert exp["parameters"]["split"]["strategy"] == "random_holdout"
        assert exp["manifest"]["kind"].startswith("loop_")
        # A rejected candidate was never trained, so it has no metrics; the others carry the
        # business metrics (#93): base rate, PR-AUC and lift at the top 10%.
        if exp["decision"] == "reject":
            assert not exp["metrics"]
            continue
        part = "val_" if exp["id"] == champion["id"] else ""
        m = exp["metrics"]
        assert 0.2 < m[f"{part}base_rate"] < 0.35  # about 26.5% churn in this sample
        assert m[f"{part}pr_auc"] > m[f"{part}base_rate"]  # beats scoring at random
        assert m[f"{part}lift_at_10pct"] > 1
    # Only the final champion has test metrics (unprefixed keys are the test metrics, as before).
    assert champion["metrics"]["pr_auc"] > champion["metrics"]["base_rate"]
    assert champion["metrics"]["test_pr_auc"] == champion["metrics"]["pr_auc"]
    assert champion["metrics"]["test_pr_auc"] > champion["metrics"]["test_base_rate"]
    assert {"f1", "accuracy", "precision", "recall", "roc_auc", "threshold"} <= set(
        champion["metrics"]
    )
    for e in exps:
        if e["id"] != champion["id"]:
            assert not any(k.startswith("test_") for k in (e["metrics"] or {})), e["id"]

    baseline = next(e for e in exps if e["parent_id"] is None)

    # Keep/reject comes from the acceptance rule, is stored on the experiment, and the
    # LLM only explains it.
    for info in result["experiments"]:
        exp = by_id[info["id"]]
        assert info["decision_mode"] == "rule"
        acceptance = exp["parameters"]["acceptance"]
        assert acceptance["rule"]["validation"] == "random"
        assert info["decision"] == ("keep" if acceptance["accepted"] else "reject")
        assert exp["decision"] == info["decision"]
        assert exp["decision_reason"].startswith(("Accepted by rule", "Rejected by rule"))
        assert len(acceptance["base_scores"]) == 15

    # Every prompt of this run is in the prompt log, and none holds a text value of the file (#48).
    calls = (await client.get(f"{base}/llm-calls")).json()
    assert [c["purpose"] for c in calls].count("agent.explain_decision") == 2
    frame = pd.read_csv(SAMPLE)
    values = {
        str(v)
        for col in frame.select_dtypes(exclude="number")
        for v in frame[col].dropna().unique()
    }
    values = {v for v in values if len(v) >= 5 and not v.replace(".", "").isdigit()}
    assert {"Month-to-month", "Fiber optic", "7590-VHVEG"} <= values
    for call in calls:
        detail = (await client.get(f"{base}/llm-calls/{call['id']}")).json()
        sent = detail["system"] + detail["prompt"]
        assert [v for v in values if v in sent] == [], call["purpose"]

    resp = await client.get(f"{base}/experiments/{baseline['id']}/export")
    assert resp.status_code == 200, resp.text

    script = tmp_path / "train.py"
    script.write_text(resp.json()["script"])
    run = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    preds_line = next(
        line for line in run.stdout.splitlines() if line.startswith("Sample predictions:")
    )
    assert "'Yes'" in preds_line or "'No'" in preds_line
    assert "0" not in preds_line.split(":", 1)[1] and "1" not in preds_line.split(":", 1)[1]


async def follow_job(client: httpx.AsyncClient, base: str, job_id: str) -> tuple[dict, list[dict]]:
    """Poll the job's events (after the last seq seen) until it finishes, like the UI does."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 300
    events: list[dict] = []
    while True:
        after = events[-1]["seq"] if events else 0
        resp = await client.get(f"{base}/jobs/{job_id}/events", params={"after": after})
        assert resp.status_code == 200, resp.text
        events += resp.json()
        status = (await client.get(f"{base}/jobs/{job_id}")).json()
        if status["status"] in ("succeeded", "failed", "cancelled") or loop.time() > deadline:
            rest = await client.get(f"{base}/jobs/{job_id}/events", params={"after": after})
            seen = {e["seq"] for e in events}
            events += [e for e in rest.json() if e["seq"] not in seen]
            return dict(status), events
        await asyncio.sleep(0.2)


async def test_explanation_failure_does_not_change_the_decision(
    client, project_id, monkeypatch, real_formulas
):
    gateway = stub_gateway()

    async def broken_complete(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(gateway, "complete_result", broken_complete)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    base = f"{API}/projects/{project_id}"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 1},
    )
    status, _ = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "succeeded", status
    info = status["result"]["experiments"][0]
    assert info["decision_mode"] == "rule"
    assert info["explanation_mode"] == "fallback"
    assert info["decision"] == ("keep" if info["acceptance"]["accepted"] else "reject")
