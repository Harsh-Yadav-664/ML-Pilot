"""A relational run through the API and the job runner (#58)."""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

import app.db.session as db_session
from app.db.models import Experiment, Feature, Job, JobEvent, Run
from app.jobs import handlers
from app.jobs.runner import JobCancelled
from app.services import run_service
from ml.agents import run_loop
from ml.experiments.acceptance import DEFAULT_RULE
from tests.fixtures.api import API
from tests.integration.test_baseline_run_api import TABLES, connection, started_run  # noqa: F401
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer

GOOD = ["good_refunds_14d", "good_cancelled_share", "good_ticket_count_60d"]


def runs_url(project_id: str, tail: str = "") -> str:
    return f"{API}/projects/{project_id}/runs{tail}"


async def finished(client: httpx.AsyncClient, project_id: str, job_id: str) -> dict[str, Any]:
    for _ in range(1200):
        job = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}")).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job  # type: ignore[no-any-return]
        await asyncio.sleep(0.5)
    raise AssertionError("the job did not finish")


@pytest.fixture
def test_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many rows each scoring of the test rows was given: the spy."""
    calls: list[int] = []
    real = run_loop.score_test

    def spy(model: Any, features: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        calls.append(len(y))
        return real(model, features, y)

    monkeypatch.setattr(run_loop, "score_test", spy)
    return calls


async def test_a_full_run_with_the_offline_stub_completes_and_scores_the_test_rows_once(
    client: httpx.AsyncClient, project_id: str, connection: str, test_calls: list[int]
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)

    started = await client.post(runs_url(project_id, f"/{run_id}/start"))
    assert started.status_code == 202, started.text
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert (await client.post(runs_url(project_id, f"/{run_id}/start"))).status_code == 409

    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert (
        state["status"] == "completed" and state["stop_reason"] == "no_llm"
    )  # the stub proposes nothing
    assert state["rounds"] == 0 and state["accepted"] == 0 and state["error"] is None
    assert state["test_metrics"] is not None and state["test_error"] is None
    assert state["champion_val_metrics"]["pr_auc"] > state["champion_val_metrics"]["base_rate"]
    assert len(test_calls) == 1 and test_calls[0] == state["test_metrics"]["n_test"]
    print(
        f"\nrun {state['status']} ({state['stop_reason']}): validation PR-AUC "
        f"{state['champion_val_metrics']['pr_auc']:.4f}, test PR-AUC {state['test_metrics']['pr_auc']:.4f} "
        f"(base rate {state['test_metrics']['base_rate']:.4f}); test rows scored {len(test_calls)} time(s)"
    )

    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None and run.finished_at is not None
        final = await db.get(Experiment, run.champion_experiment_id)
        assert final is not None and final.manifest["kind"] == "final" and final.test_metrics
        baseline = await db.get(Experiment, final.parent_id)
        assert baseline is not None and baseline.manifest["kind"] == "baseline"
        assert baseline.test_metrics is None  # only the final model ever saw the test rows
        events = (
            await db.scalars(
                select(JobEvent).where(JobEvent.job_id == job["id"]).order_by(JobEvent.seq)
            )
        ).all()
        kinds = [e.type for e in events]
        assert "baseline" in kinds and kinds[-1] == "run_finished"


async def test_a_scripted_model_run_records_every_proposal_and_its_decision(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    test_calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Costly([answer(k) for k in GOOD] + [answer("leaky_sql")], cost=0.01)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    started = await client.post(
        runs_url(project_id, f"/{run_id}/start"),
        json={"max_rounds": 4, "patience": 4, "max_cost_usd": 0.5},
    )
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job

    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "completed" and state["stop_reason"] in ("max_rounds", "patience")
    assert state["rounds"] == 4 and state["budget"]["max_cost_usd"] == 0.5
    assert state["budget_used"]["cost_usd"] == pytest.approx(
        0.05
    )  # the rejected one was asked twice
    assert len(test_calls) == 1
    feats = (await client.get(runs_url(project_id, f"/{run_id}/features"))).json()["features"]
    proposals = [f for f in feats if f["kind"] == "llm_sql"]
    assert [f["name"] for f in proposals] == [
        "refund_count_14d",
        "cancelled_order_share_180d",
        "ticket_count_60d",
        "unnamed_proposal_4",  # the repair round got no usable answer
    ]
    *tested, refused = proposals
    assert {f["status"] for f in tested} <= {"accepted", "rejected_gain"}
    # free SQL is off, so the fourth was refused, sent back once, and never reached the gain test
    assert refused["status"] == "rejected_guard" and refused["gain"] is None
    assert refused["guard_results"]["repaired"] and refused["guard_results"]["status"] == "invalid"
    assert refused["guard_results"]["attempts"][0]["stage"] == "schema"
    for f in tested:
        gain = f["gain"]
        assert gain["rule"]["validation"] == "temporal" and len(gain["base_scores"]) == len(
            gain["candidate_scores"]
        )
        assert f["status"] == ("accepted" if gain["accepted"] else "rejected_gain")
        assert gain["accepted"] == (gain["mean_gain"] > gain["margin"])  # the rule, nothing else
        assert f["sql"] and "l.cutoff_time" in f["sql"]
    assert state["accepted"] == sum(f["status"] == "accepted" for f in tested)
    print(
        "\n"
        + ", ".join(f"{f['name']}: {f['status']} ({f['gain']['mean_gain']:+.4f})" for f in tested)
    )


class FakeCtx:
    """A job context that reports a cancel request from the n-th check on."""

    def __init__(self, cancel_from_check: int) -> None:
        self.checks = 0
        self.cancel_from = cancel_from_check
        self.events: list[str] = []

    async def step(self, name: str, progress: float | None = None, **payload: Any) -> None:
        return None

    async def emit(self, type_: str, **payload: Any) -> int:
        self.events.append(type_)
        return len(self.events)

    async def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks >= self.cancel_from:
            raise JobCancelled("cancelled by request")


async def test_cancelling_mid_run_leaves_a_consistent_run_with_its_last_champion(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    test_calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)  # accept any gain: the champion moves
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in GOOD], cost=0.01)
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    ctx = FakeCtx(cancel_from_check=3)  # rounds 1 and 2 run, the check before round 3 cancels
    with pytest.raises(JobCancelled):
        await run_service.execute(
            ctx,  # type: ignore[arg-type]
            {"run_id": run_id, "request": {"max_rounds": 3, "patience": 3}},
            lambda: gateway,
        )
    assert test_calls == []  # the test rows were never scored

    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "cancelled" and "champion" in state["error"]
    assert state["rounds"] == 2 and state["accepted"] == 2
    assert state["test_metrics"] is None and state["finished_at"] is not None
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        champion = await db.get(Experiment, run.champion_experiment_id)
        assert champion is not None and champion.manifest["kind"] == "champion"
        assert champion.manifest["round"] == 2 and champion.test_metrics is None
        parent = await db.get(Experiment, champion.parent_id)
        assert parent is not None and parent.manifest["round"] == 1
        assert champion.feature_set[-2:] == ["refund_count_14d", "cancelled_order_share_180d"]
        accepted = (
            await db.scalars(
                select(Feature).where(
                    Feature.run_id == run_id,
                    Feature.status == "accepted",
                    Feature.kind == "llm_sql",
                )
            )
        ).all()
        assert [f.name for f in accepted] == ["refund_count_14d", "cancelled_order_share_180d"][
            : len(accepted)
        ]
        assert len(accepted) == 2
        assert (
            await db.scalar(select(Job).where(Job.run_id == run_id))
        ) is None  # not via the queue


async def test_a_run_needs_a_snapshot_and_a_known_id(
    client: httpx.AsyncClient, project_id: str
) -> None:
    unknown = await client.post(runs_url(project_id, f"/{uuid.uuid4()}/start"))
    assert unknown.status_code == 404
