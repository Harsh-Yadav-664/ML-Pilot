"""Optional MLflow logging of a finished run (#64).

Set ``MLFLOW_TRACKING_URI`` and, when a run ends, its parameters, metrics, champion features
(one SQL file each), model, manifest, task and evidence report are logged as one MLflow run.

* ``mlflow`` is an optional extra (``requirements-mlflow.txt``). This module imports without it;
  with the URI set and mlflow missing, :func:`log_run` returns a ``failed`` result that says so.
* Logging never changes or fails the run: :func:`log_run` never raises, and what it did is
  returned as a :class:`LogResult` for the caller to record (AGENTS.md rule 8: failures are loud).
* A tracking URI can carry a password (``https://user:secret@host``); it is never logged or
  returned, only :func:`redacted`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

ENV_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT = "MLFLOW_EXPERIMENT_NAME"
MAX_PARAM_CHARS = 500  # MLflow rejects longer parameter values on some backends
RUN_ID_TAG = "mlpilot.run_id"


def tracking_uri() -> str | None:
    """The configured tracking URI, or None when MLflow logging is off."""
    return os.environ.get(ENV_URI, "").strip() or None


def redacted(uri: str) -> str:
    """The URI without user, password or query: safe to log."""
    parts = urlsplit(uri)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@dataclass
class MlflowRunData:
    """What is logged. Plain data: building it is the caller's job, sending it is this module's."""

    run_id: str
    task_name: str
    params: dict[str, Any]
    metrics: dict[str, float | None]  # None or non-finite values are skipped
    features: list[tuple[str, str]]  # (name, sql) of the champion's features
    manifest: dict[str, Any]
    task_yaml: str
    report_html: str | None = None
    model_file: Path | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LogResult:
    status: Literal["logged", "failed"]
    message: str
    mlflow_run_id: str | None = None
    experiment: str | None = None
    uri: str | None = None  # redacted

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "mlflow_run_id": self.mlflow_run_id,
            "experiment": self.experiment,
            "tracking_uri": self.uri,
        }


def _clean_metrics(metrics: dict[str, float | None]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    }


def _write_files(root: Path, data: MlflowRunData) -> None:
    """The artifacts as files, laid out the way they appear in MLflow."""
    (root / "features").mkdir()
    for name, sql in data.features:
        (root / "features" / f"{name}.sql").write_text(sql.rstrip() + "\n")
    (root / "manifest.json").write_text(json.dumps(data.manifest, indent=2, default=str))
    (root / "task.yaml").write_text(data.task_yaml)
    if data.report_html is not None:
        (root / "report.html").write_text(data.report_html)


def log_run(data: MlflowRunData, uri: str) -> LogResult:
    """Log ``data`` to the MLflow server or store at ``uri``. Never raises."""
    shown = redacted(uri)
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param
    except ImportError:
        message = (
            f"{ENV_URI} is set ({shown}) but the optional 'mlflow' package is not installed, "
            "so nothing was logged. Install it with: pip install -r requirements-mlflow.txt"
        )
        log.warning(message)
        return LogResult("failed", message, uri=shown)
    try:
        client = MlflowClient(tracking_uri=uri)
        experiment_name = os.environ.get(ENV_EXPERIMENT) or f"mlpilot/{data.task_name}"
        found = client.get_experiment_by_name(experiment_name)
        experiment_id = found.experiment_id if found else client.create_experiment(experiment_name)
        run = client.create_run(
            experiment_id,
            run_name=f"{data.task_name}-{data.run_id[:8]}",
            tags={RUN_ID_TAG: data.run_id, **data.tags},
        )
        mlflow_run_id = run.info.run_id
        status = "FAILED"
        try:
            client.log_batch(
                mlflow_run_id,
                params=[
                    Param(k, str(v)[:MAX_PARAM_CHARS])
                    for k, v in data.params.items()
                    if v is not None
                ],
                metrics=[Metric(k, v, 0, 0) for k, v in _clean_metrics(data.metrics).items()],
            )
            with tempfile.TemporaryDirectory(prefix="mlpilot_mlflow_") as tmp:
                _write_files(Path(tmp), data)
                client.log_artifacts(mlflow_run_id, tmp)
            if data.model_file is not None and data.model_file.exists():
                client.log_artifact(mlflow_run_id, str(data.model_file), "model")
            status = "FINISHED"
        finally:
            client.set_terminated(mlflow_run_id, status)
    except Exception as e:  # noqa: BLE001 - any MLflow or network error becomes a failed LogResult; the run is untouched
        message = f"Logging to MLflow ({shown}) failed: {type(e).__name__}: {e}".replace(uri, shown)
        log.warning(message)
        return LogResult("failed", message, uri=shown)
    return LogResult(
        "logged",
        f"Logged to MLflow experiment '{experiment_name}' as run {mlflow_run_id}",
        mlflow_run_id=mlflow_run_id,
        experiment=experiment_name,
        uri=shown,
    )
