"""Data versions of a connected database through the API (#97), and the run manifest (#94).

Covers: a snapshot is a job with progress and a result; the same data is the same version;
a live version is immediate and says it cannot be reproduced; a run whose data version is a
live version records ``as_of``, counts and latest event times in its manifest.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

import app.db.session as db_session
from app.core import datasets
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from ml.experiments.manifest import RunManifest
from tests.fixtures.api import API, load_sample
from tests.fixtures.demo_db import demo_sqlite

TABLES = ["customers", "orders"]
AS_OF = "2024-06-30T12:00:00Z"


@pytest.fixture
def demo_path(tmp_path: Path) -> Path:
    return demo_sqlite(tmp_path / "demo.sqlite")


@pytest.fixture
async def connection(client: httpx.AsyncClient, project_id: str, demo_path: Path) -> str:
    resp = await client.post(
        f"{API}/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "sqlite", "database": str(demo_path)},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def take(
    client: httpx.AsyncClient, project_id: str, connection: str, **body: Any
) -> httpx.Response:
    return await client.post(
        f"{API}/projects/{project_id}/connections/{connection}/snapshots",
        json={"tables": TABLES, "as_of": AS_OF, **body},
    )


async def finish(client: httpx.AsyncClient, project_id: str, job_id: str) -> dict[str, Any]:
    url = f"{API}/projects/{project_id}/jobs/{job_id}"
    for _ in range(600):
        job = (await client.get(url)).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return dict(job)
        await asyncio.sleep(0.1)
    raise AssertionError(f"snapshot job did not finish: {job}")


async def snapshot_id(client: httpx.AsyncClient, project_id: str, connection: str) -> str:
    resp = await take(client, project_id, connection)
    assert resp.status_code == 202, resp.text
    job = await finish(client, project_id, resp.json()["job"]["id"])
    assert job["status"] == "succeeded", job
    return str(job["result"]["data_version_id"])


async def test_a_snapshot_is_a_job_with_steps_and_a_listed_version(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    resp = await take(client, project_id, connection)
    assert resp.status_code == 202 and resp.json()["version"] is None
    job_id = resp.json()["job"]["id"]
    job = await finish(client, project_id, job_id)
    assert job["status"] == "succeeded", job
    events = (await client.get(f"{API}/projects/{project_id}/jobs/{job_id}/events")).json()
    steps = [e["payload"]["name"] for e in events if e["type"] == "step"]
    assert "Reading customers" in steps and "Reading orders" in steps and "Snapshot ready" in steps

    version_id = job["result"]["data_version_id"]
    listed = (await client.get(f"{API}/projects/{project_id}/db-versions")).json()
    assert [v["id"] for v in listed] == [version_id]
    v = (await client.get(f"{API}/projects/{project_id}/db-versions/{version_id}")).json()
    assert v["mode"] == "snapshot" and v["reproducible"] is True and v["kind"] == "db_snapshot"
    assert v["short_hash"] == version_id[:12] and v["as_of"].startswith("2024-06-30T12:00:00")
    assert v["connection"]["name"] == "shop"
    assert set(v["tables"]) == set(TABLES) and v["tables"]["orders"]["rows"] > 1000
    assert v["n_rows"] == sum(t["rows"] for t in v["tables"].values())
    print(f"\nsnapshot {v['short_hash']}: {v['n_rows']} rows, steps {steps}")


async def test_the_same_data_is_the_same_version_and_a_new_order_is_another(
    client: httpx.AsyncClient, project_id: str, connection: str, demo_path: Path
) -> None:
    first = await snapshot_id(client, project_id, connection)
    again = await snapshot_id(client, project_id, connection)
    assert first == again
    assert len((await client.get(f"{API}/projects/{project_id}/db-versions")).json()) == 1

    with sqlite3.connect(demo_path) as con:
        con.execute(
            "INSERT INTO orders VALUES (900001, 1, '2024-06-01 10:00:00', 'completed', 9.99)"
        )
    changed = await snapshot_id(client, project_id, connection)
    assert changed != first
    assert len((await client.get(f"{API}/projects/{project_id}/db-versions")).json()) == 2


async def test_live_mode_answers_at_once_and_says_it_cannot_be_reproduced(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    resp = await take(client, project_id, connection, mode="live")
    assert resp.status_code == 201, resp.text
    v = resp.json()["version"]
    assert resp.json()["job"] is None
    assert v["mode"] == "live" and v["kind"] == "db_live" and v["reproducible"] is False
    assert "cannot be reproduced exactly" in v["note"]
    assert v["tables"]["orders"]["max_event_time"] <= "2024-06-30 12:00:00"
    assert (
        await client.get(f"{API}/projects/{project_id}/db-versions/{v['id']}")
    ).status_code == 200


async def test_a_run_on_a_live_version_records_as_of_counts_and_latest_event_times_in_its_manifest(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    live = (await take(client, project_id, connection, mode="live")).json()["version"]
    sample = await load_sample(client, project_id)
    async with db_session.AsyncSessionLocal() as db:
        service = ExperimentService(db)
        exp = await service.create(
            ExperimentCreate(
                project_id=project_id,
                dataset_version=str(datasets.VERSIONS_DIR / f"{sample['data_version_id']}.csv"),
                data_version_id=live["id"],
                hypothesis="baseline",
                change_description="manifest check",
                model_name="RandomForestClassifier",
                feature_set=[],
                parameters={
                    "target_column": "Churn",
                    "model_params": {"n_estimators": 10, "random_state": 1},
                },
            )
        )
        await db.commit()
    await ExperimentService(None).run_experiment_background(exp.id)

    async with db_session.AsyncSessionLocal() as db:
        stored = await ExperimentService(db).get(exp.id)
    assert stored is not None and stored.status == "completed", stored and stored.decision_reason
    manifest = RunManifest.model_validate(stored.manifest)
    data = manifest.data
    assert data is not None and data.mode == "live" and data.reproducible is False
    assert data.as_of.isoformat().startswith("2024-06-30T12:00:00")
    assert data.tables["orders"].rows == live["tables"]["orders"]["rows"] > 1000
    assert data.tables["customers"].rows == live["tables"]["customers"]["rows"]
    assert data.max_event_times["orders"] == live["tables"]["orders"]["max_event_time"]
    assert data.max_event_times["customers"] is not None
    assert "cannot be reproduced exactly" in (data.note or "")
    assert manifest.data_version_id == live["id"]
    dumped = stored.manifest["data"]
    print(
        f"\nmanifest.data: mode={dumped['mode']} as_of={dumped['as_of']} "
        f"counts={ {k: t['rows'] for k, t in dumped['tables'].items()} } "
        f"max_event_time={ {k: t['max_event_time'] for k, t in dumped['tables'].items()} }"
    )


async def test_a_file_run_has_no_database_description(
    client: httpx.AsyncClient, project_id: str
) -> None:
    sample = await load_sample(client, project_id)
    async with db_session.AsyncSessionLocal() as db:
        exp = await ExperimentService(db).create(
            ExperimentCreate(
                project_id=project_id,
                dataset_version=str(datasets.VERSIONS_DIR / f"{sample['data_version_id']}.csv"),
                hypothesis="baseline",
                change_description="file run",
                model_name="RandomForestClassifier",
                feature_set=[],
                parameters={
                    "target_column": "Churn",
                    "model_params": {"n_estimators": 10, "random_state": 1},
                },
            )
        )
        await db.commit()
    await ExperimentService(None).run_experiment_background(exp.id)
    async with db_session.AsyncSessionLocal() as db:
        stored = await ExperimentService(db).get(exp.id)
    assert stored is not None and stored.manifest is not None
    assert RunManifest.model_validate(stored.manifest).data is None


async def test_bad_requests_fail_loudly(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    unknown = await take(client, project_id, connection, tables=["nope"], mode="live")
    assert unknown.status_code == 422 and "nope" in unknown.json()["detail"]
    future = await take(client, project_id, connection, as_of="2999-01-01T00:00:00Z")
    assert future.status_code == 422 and "in the future" in future.json()["detail"]
    missing = await take(client, project_id, "no-such-connection")
    assert missing.status_code == 404
    other = (
        await client.post(
            f"{API}/projects/", json={"name": "other", "task_type": "binary_classification"}
        )
    ).json()["id"]
    assert (
        await take(client, other, connection)
    ).status_code == 404  # not this project's connection
    assert (await client.get(f"{API}/projects/{project_id}/db-versions/none")).status_code == 404


async def test_a_failed_snapshot_is_a_failed_job_with_the_reason(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    resp = await take(client, project_id, connection, tables=["nope"])  # unknown table
    assert resp.status_code == 202  # accepted; the job reports the failure
    job = await finish(client, project_id, resp.json()["job"]["id"])
    assert job["status"] == "failed" and "nope" in job["error"]
