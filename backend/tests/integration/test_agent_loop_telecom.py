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
    status = await wait_for_job(client, base, job.json()["job_id"])
    assert status["status"] == "completed", status
    result = status["result"]

    assert len(result["experiments"]) == 2
    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    assert len(tree) == 3  # baseline + 2 iterations
    exps = [(await client.get(f"{base}/experiments/{n['id']}")).json() for n in tree]
    for exp in exps:
        assert exp["status"] == "completed", exp["decision_reason"]
        assert exp["metrics"] and "f1" in exp["metrics"]
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


async def wait_for_job(client: httpx.AsyncClient, base: str, job_id: str) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 300
    while True:
        body = (await client.get(f"{base}/agent/auto-optimize/{job_id}")).json()
        if body["status"] != "running" or loop.time() > deadline:
            return dict(body)
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
