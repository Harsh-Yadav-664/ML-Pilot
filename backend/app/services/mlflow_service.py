"""Log a finished run to MLflow when ``MLFLOW_TRACKING_URI`` is set (#64).

The pieces are the ones the report and the export already produce: ``ReportService.records``
(parameters, metrics, the report), ``bundle_service.latest_features`` (the champion's SQL) and
the model file ``save_artifacts`` wrote. Nothing here may fail or change the run: every error
becomes a ``failed`` result that is logged and recorded as a ``mlflow`` job event.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core import datasets
from app.db.models import Experiment, Run
from app.jobs import runner
from app.jobs.runner import JobContext
from app.services.bundle_service import latest_features
from app.services.report_service import ReportService
from ml.export.artifacts import MODEL_FILE, artifact_dir
from ml.export.mlflow_logger import MlflowRunData, failed, log_run, tracking_uri
from ml.reports import build_report, to_html
from ml.tasks.spec import from_yaml

log = logging.getLogger(__name__)


def _prefixed(prefix: str, metrics: dict[str, Any] | None) -> dict[str, float | None]:
    return {f"{prefix}_{k}": v for k, v in (metrics or {}).items()}


async def collect(project_id: str, run_id: str) -> MlflowRunData:
    """Everything that is logged for a run that has ended, read from what the run stored."""
    async with runner.sessions()() as db:
        records = await ReportService(db).records(project_id, run_id)
        run = await db.get(Run, run_id)
        assert run is not None
        champion = (
            await db.get(Experiment, run.champion_experiment_id)
            if run.champion_experiment_id
            else None
        )
        by_name = await latest_features(db, run_id)
    spec = from_yaml(records["task"]["yaml"])
    plan = records["run"]["split_plan"]
    baseline = records["baseline"]["experiment"]
    names = list(champion.feature_set) if champion else []
    sql = [(n, str(by_name[n].sql)) for n in names if n in by_name]
    params: dict[str, Any] = {
        "run_id": run_id,
        "task": records["task"]["name"],
        "task_version": records["task"]["version"],
        "entity_table": spec.entity.table,
        "target_type": spec.target.type,
        "horizon": spec.horizon,
        "cutoffs": f"{spec.cutoffs.start}..{spec.cutoffs.end} every {spec.cutoffs.every}",
        "metric": spec.metric,
        "data_version": records["data"]["id"],
        "as_of": records["run"]["as_of"],
        "split_val_from": plan["val_from"],
        "split_test_from": plan["test_from"],
        "split_folds": plan["folds"],
        "seed": records["run"]["seed"],
        "engine": baseline.get("engine") or records["run"]["engine"],
        "engine_version": baseline.get("engine_version"),
        "n_features": len(names),
    }
    metrics = {
        **_prefixed("val", champion.val_metrics if champion else None),
        **_prefixed("test", champion.test_metrics if champion else None),
    }
    model = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id) / MODEL_FILE
    return MlflowRunData(
        run_id=run_id,
        task_name=records["task"]["name"],
        params=params,
        metrics=metrics,
        features=sql,
        manifest=records["run"]["manifest"],
        task_yaml=records["task"]["yaml"],
        report_html=to_html(build_report(records)),
        model_file=model if model.exists() else None,
    )


async def log_finished_run(ctx: JobContext, project_id: str, run_id: str) -> dict[str, Any] | None:
    """Log the run if MLflow logging is on; None if it is off. Never raises.

    The result is recorded as a ``mlflow`` event of the run's job, and logged as a warning when
    it failed.
    """
    uri = tracking_uri()
    if uri is None:
        return None
    try:
        result = await asyncio.to_thread(log_run, await collect(project_id, run_id), uri)
    except Exception as e:  # noqa: BLE001 - nothing about MLflow may fail the run; recorded as failed
        result = failed(uri, f"{type(e).__name__}: {e}")
    try:
        await ctx.emit("mlflow", **result)
    except Exception as e:  # noqa: BLE001 - the event is bookkeeping; the warning above already says what happened
        log.warning("could not record the mlflow event for run %s: %s", run_id, e)
    return result
