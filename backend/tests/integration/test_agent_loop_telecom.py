"""Baseline + agent iterations on the bundled telecom sample (string Yes/No target)."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.session as db_session
import ml.agents.decision_agent as decision_agent_module
from app.db.base import Base
from app.db.models.experiment import Experiment
from app.db.session import get_db
from app.main import app
from ml.agents.decision_agent import DecisionAgent
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


async def test_baseline_and_two_iterations_complete_and_export_decodes(
    session_factory, tmp_path, monkeypatch
):
    # Skip Optuna tuning (20 trials per run) to keep the test fast; training still runs.
    monkeypatch.setitem(sys.modules, "optuna", None)
    agent = DecisionAgent(stub_gateway(), settings=None)
    result = await agent.run_optimization_loop(str(SAMPLE), "Churn", n_hypotheses=2)

    assert len(result["experiments"]) == 2
    async with session_factory() as session:
        exps = (await session.execute(select(Experiment))).scalars().all()
    assert len(exps) == 3  # baseline + 2 iterations
    for exp in exps:
        assert exp.status == "completed", exp.decision_reason
        assert exp.metrics and "f1" in exp.metrics
        # Default positive class is the minority class ("Yes" in this sample)
        assert exp.parameters["target_encoding"] == {"classes": ["No", "Yes"], "positive_class": "Yes"}

    baseline = next(e for e in exps if e.parent_id is None)

    # Keep/reject comes from the acceptance rule, is stored on the experiment, and the
    # LLM only explains it.
    by_id = {e.id: e for e in exps}
    for info in result["experiments"]:
        exp = by_id[info["id"]]
        assert info["decision_mode"] == "rule"
        acceptance = exp.parameters["acceptance"]
        assert info["decision"] == ("keep" if acceptance["accepted"] else "reject")
        assert exp.decision == info["decision"]
        assert exp.decision_reason.startswith(("Accepted by rule", "Rejected by rule"))
        assert len(acceptance["base_scores"]) == 15

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/v1/ui/experiments/{baseline.id}/export")
    finally:
        app.dependency_overrides.clear()
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
    preds_line = next(line for line in run.stdout.splitlines() if line.startswith("Sample predictions:"))
    assert "'Yes'" in preds_line or "'No'" in preds_line
    assert "0" not in preds_line.split(":", 1)[1] and "1" not in preds_line.split(":", 1)[1]


async def test_explanation_failure_does_not_change_the_decision(session_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    gateway = stub_gateway()

    async def broken_complete(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(gateway, "complete", broken_complete)
    result = await DecisionAgent(gateway, settings=None).run_optimization_loop(str(SAMPLE), "Churn", n_hypotheses=1)
    info = result["experiments"][0]
    assert info["decision_mode"] == "rule"
    assert info["explanation_mode"] == "fallback"
    assert info["decision"] == ("keep" if info["acceptance"]["accepted"] else "reject")
