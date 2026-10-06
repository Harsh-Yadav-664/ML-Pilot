"""Baseline + agent iterations on the bundled telecom sample (string Yes/No target)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.session as db_session
import ml.agents.decision_agent as decision_agent_module
from app.db.base import Base
from ml.agents.decision_agent import DecisionAgent
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.api import API, load_sample
from tests.fixtures.gateway import stub_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
async def session_factory(tmp_path, monkeypatch):
    import app.db.models  # noqa: F401  (register tables)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    # The agent and the background runner open their own sessions.
    monkeypatch.setattr(db_session, "AsyncSessionLocal", factory)
    monkeypatch.setattr(decision_agent_module, "AsyncSessionLocal", factory)
    yield factory
    await engine.dispose()


@pytest.fixture
def real_formulas(monkeypatch):
    """The offline stub has no planner output, so propose real formulas on this sample.
    (Without this the planner falls back to a formula that the safe evaluator rejects.)"""
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
    client, project_id, tmp_path, monkeypatch, real_formulas
):
    """The Phase 0 end-to-end flow, driven only through the project API like the UI."""
    # Skip Optuna tuning (20 trials per run) to keep the test fast; training still runs.
    monkeypatch.setitem(sys.modules, "optuna", None)
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
    steps = [
        e["payload"]["name"] for e in events if e["type"] == "step" and "progress" in e["payload"]
    ]
    assert steps[-2:] == ["Hypothesis 1 of 2", "Hypothesis 2 of 2"]

    assert len(result["experiments"]) == 2
    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    assert len(tree) == 3  # baseline + 2 iterations
    exps = [(await client.get(f"{base}/experiments/{n['id']}")).json() for n in tree]
    for exp in exps:
        assert exp["status"] == "completed", exp["decision_reason"]
        assert exp["metrics"] and "f1" in exp["metrics"]
        # Business metrics (#93) on both splits: base rate, PR-AUC and lift at the top 10%.
        for part in ("val", "test"):
            m = exp["metrics"]
            assert 0.2 < m[f"{part}_base_rate"] < 0.35  # about 26.5% churn in this sample
            assert m[f"{part}_pr_auc"] > m[f"{part}_base_rate"]  # beats scoring at random
            assert m[f"{part}_lift_at_10pct"] > 1
        assert exp["parameters"]["threshold"]["chosen_on"] == "validation"
        assert exp["parameters"]["calibration"]["method"] in ("none", "isotonic", "platt")
        # Default positive class is the minority class ("Yes" in this sample)
        assert exp["parameters"]["target_encoding"] == {
            "classes": ["No", "Yes"],
            "positive_class": "Yes",
        }

    baseline = next(e for e in exps if e["parent_id"] is None)

    # Keep/reject comes from the acceptance rule, is stored on the experiment, and the
    # LLM only explains it.
    by_id = {e["id"]: e for e in exps}
    for info in result["experiments"]:
        exp = by_id[info["id"]]
        assert info["decision_mode"] == "rule"
        acceptance = exp["parameters"]["acceptance"]
        assert info["decision"] == ("keep" if acceptance["accepted"] else "reject")
        assert exp["decision"] == info["decision"]
        assert exp["decision_reason"].startswith(("Accepted by rule", "Rejected by rule"))
        assert len(acceptance["base_scores"]) == 15

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
    session_factory, monkeypatch, real_formulas
):
    monkeypatch.setitem(sys.modules, "optuna", None)
    gateway = stub_gateway()

    async def broken_complete(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(gateway, "complete", broken_complete)
    result = await DecisionAgent(gateway, settings=None).run_optimization_loop(
        str(SAMPLE), "Churn", n_hypotheses=1, project_id="p"
    )
    info = result["experiments"][0]
    assert info["decision_mode"] == "rule"
    assert info["explanation_mode"] == "fallback"
    assert info["decision"] == ("keep" if info["acceptance"]["accepted"] else "reject")
