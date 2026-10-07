"""The baseline of a run through the API (#55): features, a model, a champion, all recorded."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

import app.db.session as db_session
from app.db.models import Experiment, Feature, Run
from ml.data.schema_graph import build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.pit_guard import check
from tests.fixtures.api import API
from tests.fixtures.demo_db import demo_sqlite
from tests.integration.test_tasks_api import (
    body,
    confirmed_task,
    take_snapshot,
    url,
)
from tests.unit.test_task_spec import CHURN

TABLES = ["customers", "orders", "refunds", "sessions", "support_tickets", "marketing_emails"]


@pytest.fixture
async def connection(client: httpx.AsyncClient, project_id: str, tmp_path: Path) -> str:
    path = demo_sqlite(tmp_path / "demo.sqlite")
    resp = await client.post(
        f"{API}/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "sqlite", "database": str(path)},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def started_run(
    client: httpx.AsyncClient, project_id: str, connection: str, version: str
) -> tuple[str, str]:
    task = await confirmed_task(client, project_id, connection, CHURN, version)
    resp = await client.post(url(project_id, f"{task}/runs"), json={"data_version_id": version})
    assert resp.status_code == 201, resp.text
    return task, str(resp.json()["id"])


async def test_the_baseline_is_trained_recorded_and_made_the_champion(
    client: httpx.AsyncClient, project_id: str, connection: str, tmp_path: Path
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    task, run_id = await started_run(client, project_id, connection, version)

    resp = await client.post(url(project_id, f"{task}/runs/{run_id}/baseline"))
    assert resp.status_code == 200, resp.text
    out = resp.json()
    m = out["metrics"]
    assert m["pr_auc"] > m["base_rate"] + 0.05, m
    assert out["engine"] == "lightgbm" and out["kept"] == len(out["features"]) > 50
    assert out["split"]["n_test"] > 0 and not any(k.startswith("test") for k in m)
    assert out["dropped_counts"].get("leakage_scan") == 1  # customers.is_churned
    print(
        f"\nHTTP 200: validation PR-AUC {m['pr_auc']:.4f} vs base rate {m['base_rate']:.4f}; "
        f"{out['kept']} features kept of {out['candidates']} candidates, "
        f"dropped {out['dropped_counts']}, {out['seconds']:.1f}s"
    )

    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None and run.champion_experiment_id == out["experiment_id"]
        assert run.engine == "lightgbm" and run.status == "running"
        assert run.manifest["baseline"]["val_pr_auc"] == m["pr_auc"]
        experiment = await db.get(Experiment, out["experiment_id"])
        assert experiment is not None and experiment.run_id == run_id
        assert experiment.val_metrics == m and experiment.test_metrics is None
        assert experiment.decision == "keep" and experiment.decision_mode == "rule"
        rows = (await db.scalars(select(Feature).where(Feature.run_id == run_id))).all()
        assert len(rows) == out["kept"]
        assert all(f.kind == "dfs" and f.status == "accepted" and f.sql for f in rows)
        assert {f.name for f in rows} == set(experiment.feature_set)
        assert all((f.ir is None) == (f.guard_results["group"] == "attribute") for f in rows)

    # every stored feature is accepted again by the guard, read from the database
    path = demo_sqlite(tmp_path / "again.sqlite")
    graph = build_schema_graph(open_source(ConnectionSpec(dialect="sqlite", database=str(path))))
    assert {s["table"] for s in out["skipped_tables"]} >= {
        "order_items",
        "customer_status_snapshot",
    }
    statuses = {check(f.sql, graph, "duckdb", allow_rewrite=False).status for f in rows}
    assert statuses == {"accepted"}

    again = await client.post(url(project_id, f"{task}/runs/{run_id}/baseline"))
    assert again.status_code == 409 and "already has a baseline" in again.json()["detail"]


async def test_a_run_without_a_snapshot_cannot_have_a_baseline(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    task = await confirmed_task(client, project_id, connection, CHURN, version)
    run_id = str(uuid.uuid4())
    async with db_session.AsyncSessionLocal() as db:
        db.add(Run(id=run_id, project_id=project_id, task_spec_id=task, engine="not_selected"))
        await db.commit()
    resp = await client.post(url(project_id, f"{task}/runs/{run_id}/baseline"))
    assert resp.status_code == 422 and "needs a snapshot" in resp.json()["detail"]


async def test_an_unknown_run_is_404(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    created = await client.post(url(project_id), json=body(connection, CHURN))
    resp = await client.post(
        url(project_id, f"{created.json()['id']}/runs/{uuid.uuid4()}/baseline")
    )
    assert resp.status_code == 404
