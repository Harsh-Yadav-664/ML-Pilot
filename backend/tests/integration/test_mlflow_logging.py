"""Optional MLflow logging of a finished run (#64).

A scripted run (the pattern of test_export_bundle.exported_run) ends with MLFLOW_TRACKING_URI set
to a file store in tmp_path; the test reads the MLflow run back through MLflow's own client. The
tests that need mlflow skip without it; CI installs it from requirements-mlflow.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select

import app.db.session as db_session
import tests.integration.test_baseline_run_api as base
from app.core import datasets
from app.db.models import Experiment, Job, JobEvent, Run
from app.jobs import handlers
from ml.experiments.acceptance import DEFAULT_RULE
from ml.export.artifacts import MODEL_FILE, artifact_dir
from ml.export.mlflow_logger import ENV_URI, MlflowRunData, log_run, redacted
from tests.integration.test_run_loop_api import GOOD, finished, runs_url
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer

TABLES, started_run = base.TABLES, base.started_run
connection = base.connection  # the fixture, re-exported


async def scripted_run(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, dict[str, Any]]:
    """A run whose model proposes two features that are kept; (run id, the finished job)."""
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
    return run_id, job


async def mlflow_events(job_id: str) -> list[dict[str, Any]]:
    async with db_session.AsyncSessionLocal() as db:
        rows = (
            await db.scalars(
                select(JobEvent).where(JobEvent.job_id == job_id, JobEvent.type == "mlflow")
            )
        ).all()
    return [dict(r.payload) for r in rows]


async def test_a_finished_run_is_logged_to_a_file_mlflow_store(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    store = tmp_path / "mlruns"
    uri = f"file:{store}"
    monkeypatch.setenv(ENV_URI, uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # newer MLflow refuses file: otherwise
    run_id, job = await scripted_run(client, project_id, connection, monkeypatch)

    # the run itself is as it always is, and the job says what was logged
    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "completed" and state["error"] is None
    (event,) = await mlflow_events(job["id"])
    assert event["status"] == "logged", event
    assert job["result"]["mlflow"]["mlflow_run_id"] == event["mlflow_run_id"]

    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None and run.champion_experiment_id is not None
        final = await db.get(Experiment, run.champion_experiment_id)
        assert final is not None and final.test_metrics and final.val_metrics
        task_name = state["task_name"]
        manifest, seed, data_version = run.manifest, run.seed, run.data_version_id
        plan = run.split_plan
        features = list(final.feature_set)
    assert len(features) >= 3

    client_ = MlflowClient(tracking_uri=uri)
    found = client_.search_runs(
        [client_.get_experiment_by_name(f"mlpilot/{task_name}").experiment_id],  # type: ignore[union-attr]
        filter_string=f"tags.`mlpilot.run_id` = '{run_id}'",
    )
    assert len(found) == 1
    mlrun = found[0]
    assert mlrun.info.run_id == event["mlflow_run_id"] and mlrun.info.status == "FINISHED"

    params = mlrun.data.params
    assert params["run_id"] == run_id and params["task"] == task_name
    assert params["data_version"] == data_version and params["seed"] == str(seed)
    assert params["split_val_from"] == str(plan["val_from"])[:10]
    assert params["split_test_from"] == str(plan["test_from"])[:10]
    assert params["engine"] and params["horizon"] and params["entity_table"] == "customers"
    assert params["n_features"] == str(len(features))

    metrics = mlrun.data.metrics
    for k, v in final.val_metrics.items():
        if isinstance(v, int | float):
            assert metrics[f"val_{k}"] == pytest.approx(v)
    for k, v in final.test_metrics.items():
        if isinstance(v, int | float):
            assert metrics[f"test_{k}"] == pytest.approx(v)
    assert "val_pr_auc" in metrics and "test_pr_auc" in metrics

    def names(path: str | None = None) -> set[str]:
        return {a.path for a in client_.list_artifacts(mlrun.info.run_id, path)}

    assert {"features", "model", "manifest.json", "report.html", "task.yaml"} <= names()
    assert names("features") == {f"features/{n}.sql" for n in features}
    assert names("model") == {"model/model.txt"}
    out = Path(client_.download_artifacts(mlrun.info.run_id, "", str(tmp_path / "dl")))
    assert json.loads((out / "manifest.json").read_text()) == json.loads(
        json.dumps(manifest, default=str)
    )
    saved = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id) / MODEL_FILE
    assert (out / "model" / "model.txt").read_text() == saved.read_text()
    assert all("select" in (out / "features" / f"{n}.sql").read_text().lower() for n in features)
    assert "<script" not in (out / "report.html").read_text()
    print(
        f"\nMLflow run {mlrun.info.run_id}: {len(params)} params, {len(metrics)} metrics, "
        f"artifacts {sorted(names())}, {len(features)} feature files"
    )


async def test_without_the_uri_nothing_is_logged_and_nothing_is_recorded(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_URI, raising=False)
    _, job = await scripted_run(client, project_id, connection, monkeypatch)
    assert await mlflow_events(job["id"]) == [] and "mlflow" not in job["result"]


async def test_the_uri_set_but_mlflow_missing_is_recorded_as_failed_and_the_run_is_unchanged(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # `import mlflow` raises ImportError
    monkeypatch.setenv(ENV_URI, "https://alice:s3cret-pw@mlflow.example.com:5000/path?token=abc")
    with caplog.at_level("WARNING"):
        run_id, job = await scripted_run(client, project_id, connection, monkeypatch)
    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "completed" and state["error"] is None
    assert state["test_metrics"] is not None
    (event,) = await mlflow_events(job["id"])
    assert event["status"] == "failed" and "not installed" in event["message"]
    assert event["tracking_uri"] == "https://mlflow.example.com:5000/path"
    assert "not installed" in caplog.text
    for text in (json.dumps(event), caplog.text, json.dumps(job["result"])):
        assert "s3cret-pw" not in text and "alice" not in text and "abc" not in text


async def test_a_logging_error_is_recorded_as_failed_and_the_run_is_unchanged(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlflow")
    not_a_directory = tmp_path / "afile"
    not_a_directory.write_text("x")
    monkeypatch.setenv(ENV_URI, f"file:{not_a_directory}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    run_id, job = await scripted_run(client, project_id, connection, monkeypatch)
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        stored = await db.get(Job, job["id"])
    assert run is not None and run.status == "completed" and run.error is None
    assert stored is not None and stored.status == "succeeded"
    (event,) = await mlflow_events(job["id"])
    assert event["status"] == "failed" and "Logging to MLflow" in event["message"], event


def test_the_error_text_never_carries_the_credentials_of_the_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mlflow")
    import mlflow

    uri = "https://alice:s3cret-pw@mlflow.example.com:5000"

    class Broken:
        def __init__(self, tracking_uri: str) -> None:
            raise RuntimeError(f"cannot reach {tracking_uri}")

    monkeypatch.setattr(mlflow, "MlflowClient", Broken)
    data = MlflowRunData("r" * 36, "t", {}, {}, [], {}, "x: 1")
    result = log_run(data, uri)
    assert result.status == "failed" and "cannot reach https://mlflow.example.com:5000" in (
        result.message
    )
    assert "s3cret-pw" not in result.message and "alice" not in result.message


def test_redacted_keeps_scheme_host_and_path_only() -> None:
    assert redacted("https://u:pw@h.example:5000/x?token=1") == "https://h.example:5000/x"
    assert redacted("file:/tmp/mlruns").endswith("/tmp/mlruns")
