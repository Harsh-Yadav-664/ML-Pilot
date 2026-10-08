"""The export bundle of a finished run (#61).

A scripted run is exported; the bundle is unzipped next to a DuckDB file holding the same
tables, and its own score.py recomputes the validation PR-AUC from that database. With
MLPILOT_EXPORT_DIR set the unzipped bundle and the database are left there for the CI job that
installs the bundle in a clean virtualenv and runs dbt compile on it.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import duckdb
import httpx
import pytest
import yaml

import app.db.session as db_session
import tests.integration.test_baseline_run_api as base
from app.core import datasets
from app.db.models import Connection, Run, TaskSpec
from app.jobs import handlers
from app.services.baseline_service import load_world
from app.services.connection_service import ConnectionService
from ml.data.workspace import workspace_for
from ml.experiments.acceptance import DEFAULT_RULE
from ml.features.baseline import flatten_graph, typed_times
from ml.tasks.spec import from_yaml
from tests.fixtures.api import API
from tests.integration.test_run_loop_api import GOOD, finished, runs_url
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer

TABLES, started_run = base.TABLES, base.started_run
connection = base.connection  # the fixture, re-exported


async def run_script(
    root: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-I", "score.py", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


async def exported_run(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, str]:
    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in GOOD[:2]], cost=0.01)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    started = await client.post(
        runs_url(project_id, f"/{run_id}/start"), json={"max_rounds": 2, "patience": 2}
    )
    assert started.status_code == 202, started.text
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job

    response = await client.get(runs_url(project_id, f"/{run_id}/export"))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    out = Path(os.environ.get("MLPILOT_EXPORT_DIR") or tmp_path / "export")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(out)
    (root,) = [p for p in out.iterdir() if p.name.startswith("mlpilot_export_")]

    # the user's database: the same tables under their own names
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None and run.task_spec_id is not None
        spec_row = await db.get(TaskSpec, run.task_spec_id)
        assert spec_row is not None and spec_row.connection_id is not None
        conn = await db.get(Connection, spec_row.connection_id)
        assert conn is not None
        graph = await ConnectionService(db).schema_graph(conn)
        spec = from_yaml(spec_row.yaml)
        task_id = spec_row.id
    world = load_world(workspace_for(project_id, datasets.PROJECTS_DIR), graph, version, task_id)
    flat, _ = flatten_graph(world.graph, spec.entity.table)
    db_path = out / "data.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    for key, table in typed_times(world.tables, flat).items():
        con.register("t", table)
        con.execute(f'CREATE TABLE "{key}" AS SELECT * FROM t')
        con.unregister("t")
    con.close()
    return root, db_path, run_id


async def test_the_bundle_reproduces_the_validation_metric_from_the_database(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, db_path, _ = await exported_run(client, project_id, connection, monkeypatch, tmp_path)
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    for needed in (
        "README.md", "task.yaml", "manifest.json", "report.html", "score.py", "sqlcheck.py",
        "requirements.txt", "entities.sql", "model/model.txt", "reference_validation.csv",
        "dbt/dbt_project.yml", "dbt/models/entities.sql", "dbt/models/features_all.sql",
    ):  # fmt: skip
        assert needed in names, (needed, sorted(names))
    features = [n for n in names if n.startswith("features/") and n.endswith(".sql")]
    assert len(features) >= 3 and len(
        [n for n in names if n.startswith("dbt/models/features/")]
    ) == len(features)
    assert yaml.safe_load((root / "task.yaml").read_text())["entity"]["table"]
    assert "<script" not in (root / "report.html").read_text()

    env = {**os.environ, "MLPILOT_DB_URL": f"duckdb:///{db_path}"}
    verified = await run_script(
        root, env, "--verify"
    )  # fmt: skip
    print(verified.stdout, verified.stderr)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "OK: reproduced within 1e-6" in verified.stdout

    scored = await run_script(
        root, env, "--cutoff", "2024-10-01", "--out", "scores.csv"
    )  # fmt: skip
    assert scored.returncode == 0, scored.stdout + scored.stderr
    lines = (root / "scores.csv").read_text().splitlines()
    assert lines[0] == "entity_id,cutoff_time,score" and len(lines) > 100
    assert all(0.0 <= float(x.split(",")[-1]) <= 1.0 for x in lines[1:])


async def test_an_unfinished_run_cannot_be_exported(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    assert (await client.get(runs_url(project_id, f"/{run_id}/export"))).status_code == 409
    assert (await client.get(runs_url(project_id, "/nope/export"))).status_code == 404


async def test_the_bundle_reproduces_the_validation_metric_on_the_demo_postgres(
    client: httpx.AsyncClient,
    project_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real thing: a Postgres source, features transpiled to Postgres, score.py through
    pg8000 as the read-only role. Runs where the compose demo database is up (CI: demo-db)."""
    host = os.environ.get("MLPILOT_DEMO_PG_HOST")
    if not host:
        if os.environ.get("MLPILOT_DEMO_PG_REQUIRED"):
            pytest.fail("MLPILOT_DEMO_PG_REQUIRED is set but MLPILOT_DEMO_PG_HOST is not")
        pytest.skip(
            "no demo Postgres: set MLPILOT_DEMO_PG_HOST, _PORT, _DB, _RO_USER, _RO_PASSWORD"
        )
    port, db = os.environ.get("MLPILOT_DEMO_PG_PORT", "5432"), os.environ["MLPILOT_DEMO_PG_DB"]
    user, password = (
        os.environ["MLPILOT_DEMO_PG_RO_USER"],
        os.environ["MLPILOT_DEMO_PG_RO_PASSWORD"],
    )
    monkeypatch.setenv("DEMO_EXPORT_TEST_PW", password)
    created = await client.post(
        f"{API}/projects/{project_id}/connections/",
        json={
            "name": "demo",
            "dialect": "postgres",
            "host": host,
            "port": int(port),
            "database": db,
            "username": user,
            "password_env": "DEMO_EXPORT_TEST_PW",
        },
    )
    assert created.status_code in (200, 201), created.text
    conn_id = created.json()["id"]

    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in GOOD[:2]], cost=0.01)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, conn_id, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, conn_id, version)
    started = await client.post(
        runs_url(project_id, f"/{run_id}/start"), json={"max_rounds": 2, "patience": 2}
    )
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job

    response = await client.get(runs_url(project_id, f"/{run_id}/export"))
    assert response.status_code == 200, response.text
    out = Path(os.environ.get("MLPILOT_EXPORT_PG_DIR") or tmp_path / "export_pg")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(out)
    (root,) = [p for p in out.iterdir() if p.name.startswith("mlpilot_export_")]
    assert '"dialect": "postgres"' in (root / "bundle.json").read_text()

    env = {**os.environ, "MLPILOT_DB_URL": f"postgresql://{user}:{password}@{host}:{port}/{db}"}
    verified = await run_script(root, env, "--verify")
    print(verified.stdout)
    assert verified.returncode == 0, verified.stdout + verified.stderr.replace(password, "***")
    assert "OK: reproduced within 1e-6" in verified.stdout
    assert password not in verified.stdout + verified.stderr  # the URL is never printed
