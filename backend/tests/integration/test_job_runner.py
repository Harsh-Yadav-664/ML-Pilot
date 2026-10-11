"""The durable job runner: restart recovery, cancellation and the ordered event log (#91)."""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.models import Experiment, Job
from app.jobs import runner
from app.jobs.runner import JobContext, handler
from app.main import app
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.api import API

FINAL = ("succeeded", "failed", "cancelled")


@handler("test_sleep")
async def _sleep_forever(ctx: JobContext, params: dict[str, Any]) -> dict[str, Any]:
    await ctx.step("working", 0.5, experiment_id=params.get("experiment_id"))
    await asyncio.sleep(3600)
    return {}


@handler("test_count")
async def _count(ctx: JobContext, params: dict[str, Any]) -> dict[str, Any]:
    for i in range(params["n"]):
        await ctx.emit("log", i=i)
    return {"emitted": params["n"]}


@pytest.fixture
def factory(metadata_db_url):
    return async_sessionmaker(
        create_async_engine(metadata_db_url, poolclass=NullPool), expire_on_commit=False
    )


async def _enqueue(factory, project_id: str, kind: str, **params: Any) -> str:
    async with factory() as db:
        job = await runner.enqueue(db, project_id=project_id, kind=kind, params=params)
        await db.commit()
    runner.notify()
    return job.id


async def _wait(client, project_id: str, job_id: str, until, timeout: float = 120) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        body = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}")).json()
        if until(body):
            return dict(body)
        assert loop.time() < deadline, f"timed out waiting; last state {body}"
        await asyncio.sleep(0.05)


async def test_a_job_killed_mid_run_is_failed_after_restart(
    client, project_id, factory, job_worker, monkeypatch
):
    async with factory() as db:
        exp = await ExperimentService(db).create(
            ExperimentCreate(
                project_id=project_id,
                dataset_version="x.csv",
                hypothesis="h",
                change_description="c",
                model_name="RandomForestClassifier",
            )
        )
        exp.status = "running"
        await db.commit()
    job_id = await _enqueue(factory, project_id, "test_sleep", experiment_id=exp.id)
    await _wait(client, project_id, job_id, lambda j: j["current_step"] == "working")

    # Kill the worker task mid-job: nothing gets to record an outcome.
    await job_worker.stop()
    await asyncio.sleep(0.3)
    assert (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}")).json()[
        "status"
    ] == "running"  # stuck, as after a crash

    # Restart the app (its real start-up), with a short staleness window for the test.
    monkeypatch.setattr(settings, "DATABASE_URL", factory.kw["bind"].url.render_as_string(False))
    monkeypatch.setattr(settings, "JOB_STALE_SECONDS", 0.2)
    async with app.router.lifespan_context(app):
        job = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}")).json()
    assert job["status"] == "failed"
    assert job["error"] == "interrupted by restart"
    assert job["finished_at"] is not None
    async with factory() as db:
        exp_after = await db.get(Experiment, exp.id)
    assert exp_after is not None
    assert (exp_after.status, exp_after.decision_reason) == ("failed", "interrupted by restart")


async def test_a_job_with_a_fresh_heartbeat_is_not_recovered(client, project_id, factory):
    job_id = await _enqueue(factory, project_id, "test_sleep")
    await _wait(client, project_id, job_id, lambda j: j["status"] == "running")
    assert job_id not in await runner.recover_interrupted(stale_after=30)
    await client.post(f"{API}/projects/{project_id}/jobs/{job_id}/cancel")


@pytest.fixture
def endless_planner(monkeypatch):
    """Propose a new valid feature every time, so only cancelling ends the loop."""
    count = iter(range(1000))

    async def propose(self, **kwargs):
        i = next(count)
        return {
            "name": f"f_{i}",
            "formula": f"f{i % 4} * f{(i + 1) % 4}",
            "reason": "test",
            "non_redundant_reasoning": "new",
        }

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)


async def test_cancelling_auto_optimize_stops_within_one_step(
    client, project_id, endless_planner, monkeypatch
):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    df["target"] = (df["f0"] + rng.normal(size=300) > 0).astype(int)
    up = await client.post(
        f"{API}/projects/{project_id}/datasets/upload",
        files={"file": ("d.csv", df.to_csv(index=False).encode())},
    )
    version = up.json()["data_version_id"]
    job = await client.post(
        f"{API}/projects/{project_id}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "target", "n_hypotheses": 20},
    )
    job_id = job.json()["id"]
    await _wait(
        client, project_id, job_id, lambda j: (j["current_step"] or "").startswith("Round 2:")
    )

    cancel = await client.post(f"{API}/projects/{project_id}/jobs/{job_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    at_cancel = int(cancel.json()["current_step"].split()[1].rstrip(":"))  # "Round k: ..."
    assert cancel.json()["cancel_requested"] is True

    final = await _wait(client, project_id, job_id, lambda j: j["status"] in FINAL)
    assert final["status"] == "cancelled" and final["error"] is None
    events = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}/events")).json()
    started = [
        int(e["payload"]["name"].split()[1].rstrip(":"))
        for e in events
        if e["type"] == "step" and e["payload"]["name"].startswith("Round")
    ]
    # The round in progress may finish; the next one never starts.
    assert max(started) == at_cancel, started
    assert events[-1]["type"] == "log" and "Cancelled" in events[-1]["payload"]["message"]
    tree = (await client.get(f"{API}/projects/{project_id}/experiments")).json()
    assert len(tree) <= 1 + at_cancel
    assert all(e["status"] in ("completed", "rejected_invalid") for e in tree)

    again = await client.post(f"{API}/projects/{project_id}/jobs/{job_id}/cancel")
    assert again.status_code == 409


async def test_cancelling_a_queued_job_is_immediate(client, project_id, factory, job_worker):
    await job_worker.stop()  # nothing claims it
    job_id = await _enqueue(factory, project_id, "test_sleep")
    resp = await client.post(f"{API}/projects/{project_id}/jobs/{job_id}/cancel")
    assert resp.json()["status"] == "cancelled" and resp.json()["finished_at"]


async def test_events_are_ordered_and_pageable(client, project_id, factory):
    job_id = await _enqueue(factory, project_id, "test_count", n=40)
    done = await _wait(client, project_id, job_id, lambda j: j["status"] in FINAL)
    assert done["status"] == "succeeded" and done["result"] == {"emitted": 40}

    url = f"{API}/projects/{project_id}/jobs/{job_id}/events"
    events = (await client.get(url)).json()
    assert [e["seq"] for e in events] == list(range(1, 41))
    assert [e["payload"]["i"] for e in events] == list(range(40))
    assert all(a["ts"] <= b["ts"] for a, b in itertools.pairwise(events))
    page = (await client.get(url, params={"after": 35})).json()
    assert [e["seq"] for e in page] == [36, 37, 38, 39, 40]

    stream = await client.get(f"{url}/stream", params={"after": 30})
    assert stream.headers["content-type"].startswith("text/event-stream")
    ids = [int(line[4:]) for line in stream.text.splitlines() if line.startswith("id: ")]
    assert ids == list(range(31, 41))
    assert 'event: end\ndata: {"status": "succeeded"}' in stream.text
    data = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    assert data[0]["seq"] == 31 and data[0]["type"] == "log"


async def test_a_failing_job_is_failed_with_its_message(client, project_id, factory):
    @handler("test_boom")
    async def boom(ctx: JobContext, params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("bad input")

    job_id = await _enqueue(factory, project_id, "test_boom")
    done = await _wait(client, project_id, job_id, lambda j: j["status"] in FINAL)
    assert (done["status"], done["error"]) == ("failed", "ValueError: bad input")


async def test_jobs_of_another_project_are_404(client, project_id, factory):
    job_id = await _enqueue(factory, project_id, "test_count", n=1)
    other = (
        await client.post(
            f"{API}/projects/", json={"name": "o", "task_type": "binary_classification"}
        )
    ).json()["id"]
    for suffix in ("", "/events", "/events/stream"):
        assert (
            await client.get(f"{API}/projects/{other}/jobs/{job_id}{suffix}")
        ).status_code == 404
    assert (await client.post(f"{API}/projects/{other}/jobs/{job_id}/cancel")).status_code == 404


async def test_unknown_job_kind_is_refused(factory, project_id):
    async with factory() as db:
        with pytest.raises(ValueError, match="Unknown job kind"):
            await runner.enqueue(db, project_id=project_id, kind="nope", params={})


async def test_two_workers_never_run_the_same_job(client, project_id, factory):
    ids = [await _enqueue(factory, project_id, "test_count", n=3) for _ in range(6)]
    for job_id in ids:
        await _wait(client, project_id, job_id, lambda j: j["status"] in FINAL)
    async with factory() as db:
        for job_id in ids:
            job = await db.get(Job, job_id)
            assert job is not None and job.status == "succeeded"
    for job_id in ids:
        events = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}/events")).json()
        assert [e["seq"] for e in events] == [1, 2, 3]  # run once, not twice
