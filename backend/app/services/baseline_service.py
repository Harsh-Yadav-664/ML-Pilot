"""Run the baseline of a run (#55): DFS features, LightGBM, recorded as the run's first champion.

The data version must be a snapshot: the features are computed from the snapshot's copies of
the tables, never from the live database. The work runs in a worker thread and nothing is
written to the metadata database until it has finished, so a failed baseline leaves the run as
it was.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pyarrow as pa
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import Experiment, Feature, Run
from app.schemas.tasks import (
    BaselineDroppedRead,
    BaselineFeatureRead,
    BaselineRead,
    BaselineSkippedRead,
)
from app.services.connection_service import ConnectionService
from app.services.task_service import TaskService
from ml.data.engine import EngineError, arrow_to_pandas
from ml.data.schema_graph import SchemaGraph
from ml.data.workspace import Workspace, workspace_for
from ml.features.baseline import (
    BaselineError,
    BaselineResult,
    build_baseline,
    restrict_graph,
)
from ml.features.dfs import Skipped
from ml.tasks import labels
from ml.tasks.spec import TaskSpec
from ml.validation.splits import SplitError, TemporalSplitPlan

NOTES = [
    (
        "Validation rows were used to stop training early, so the validation score is slightly "
        "optimistic. The test rows were not read."
    ),
    (
        "Relational features are cutoff-safe by construction and re-checked by the point-in-time "
        "guard. Attributes of the entity row are not, so they go through the leakage scan."
    ),
]


def _load(
    workspace: Workspace,
    graph: SchemaGraph,
    version_id: str,
    task_id: str,
    spec: TaskSpec,
    plan: TemporalSplitPlan,
) -> tuple[BaselineResult, list[Skipped]]:
    refs = workspace.snapshot_tables(version_id)
    graph, absent = restrict_graph(graph, set(refs))
    tables: dict[str, pa.Table] = {
        t.name: workspace.source.read_table(refs[t.key]) for t in graph.tables
    }
    frame = arrow_to_pandas(
        workspace.source.read_table(labels.label_table_ref(task_id, version_id))
    )
    return build_baseline(spec, graph, tables, frame, plan), absent


class BaselineService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run(self, project_id: str, task_id: str, run_id: str) -> BaselineRead:
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id or run.task_spec_id != task_id:
            raise HTTPException(404, "Run not found for this task")
        if run.champion_experiment_id is not None:
            raise HTTPException(409, "This run already has a baseline; start a new run")
        if run.data_version_id is None:
            raise HTTPException(
                422, "The run was started on the live database; the baseline needs a snapshot"
            )
        tasks = TaskService(self.db)
        built = await tasks.labels_for_run(project_id, task_id, run.data_version_id)
        if built.mode != "snapshot":
            raise HTTPException(422, "The baseline needs a snapshot data version, not a live one")
        connections = ConnectionService(self.db)
        assert built.row.connection_id is not None  # _build_labels refuses a task without one
        conn = await connections.get(project_id, built.row.connection_id)
        graph = await connections.schema_graph(conn)
        plan = TemporalSplitPlan.model_validate(run.split_plan)
        workspace = workspace_for(project_id, datasets.PROJECTS_DIR)
        # the workspace is read in a worker thread; commit first so no write is held meanwhile
        await self.db.commit()
        try:
            result, absent = await asyncio.to_thread(
                _load, workspace, graph, run.data_version_id, task_id, built.spec, plan
            )
        except (BaselineError, SplitError, labels.LabelError, EngineError) as e:
            raise HTTPException(422, str(e)) from None
        run = await self.db.get(Run, run_id)
        assert run is not None
        return await self._record(run, built.spec.name, result, absent)

    async def _record(
        self, run: Run, task_name: str, result: BaselineResult, absent: list[Skipped]
    ) -> BaselineRead:
        now = datetime.now(UTC)
        experiment = Experiment(
            id=str(uuid.uuid4()),
            project_id=run.project_id,
            dataset_version=str(run.data_version_id)[:50],
            hypothesis="Baseline: automatic aggregations over the related tables, no LLM",
            change_description=(
                f"{len(result.features)} features from {result.candidates} candidates "
                f"(dropped: {result.dropped_counts() or 'none'})"
            ),
            model_name="LGBMClassifier",
            parameters=result.params,
            validation_config=result.split,
            preprocessing_config={},
            feature_set=[f.candidate.name for f in result.features],
            metrics=result.metrics,
            val_metrics=result.metrics,
            status="completed",
            decision="keep",
            decision_reason="First model of the run: the baseline every later feature is measured against",
            decision_mode="rule",
            runtime_seconds=result.seconds,
            run_id=run.id,
            data_version_id=run.data_version_id,
            manifest={
                "engine": result.engine,
                "engine_version": result.engine_version,
                "seed": result.seed,
                "split": result.split,
                "params": result.params,
                "task": task_name,
                "kind": "baseline",
            },
        )
        self.db.add(experiment)
        await self.db.flush()
        for f in result.features:
            c = f.candidate
            self.db.add(
                Feature(
                    id=str(uuid.uuid4()),
                    run_id=run.id,
                    name=c.name,
                    kind="dfs",
                    sql=c.sql,
                    ir=c.ir_json,
                    rationale=c.description,
                    status="accepted",
                    guard_results={"status": "accepted", "dialect": "duckdb", "group": c.group},
                    gain={"importance": f.importance, "metric": "lightgbm gain share"},
                )
            )
        skipped = [*result.skipped_tables, *absent]
        run.engine = result.engine
        run.status = "running"
        run.started_at = run.started_at or now
        run.champion_experiment_id = experiment.id
        run.manifest = {
            **run.manifest,
            "baseline": {
                "experiment_id": experiment.id,
                "candidates": result.candidates,
                "kept": len(result.features),
                "dropped": result.dropped_counts(),
                "skipped_tables": [{"table": s.table, "reason": s.reason} for s in skipped],
                "val_pr_auc": result.metrics["pr_auc"],
                "val_base_rate": result.metrics["base_rate"],
                "at": now.isoformat(),
            },
        }
        await self.db.flush()
        return BaselineRead(
            run_id=run.id,
            experiment_id=experiment.id,
            engine=result.engine,
            engine_version=result.engine_version,
            seed=result.seed,
            metrics=result.metrics,
            split=result.split,
            candidates=result.candidates,
            kept=len(result.features),
            dropped_counts=result.dropped_counts(),
            skipped_tables=[BaselineSkippedRead(table=s.table, reason=s.reason) for s in skipped],
            features=[
                BaselineFeatureRead(
                    name=f.candidate.name,
                    group=f.candidate.group,
                    description=f.candidate.description,
                    sql=f.candidate.sql,
                    ir=f.candidate.ir_json,
                    importance=f.importance,
                )
                for f in result.features
            ],
            dropped=[
                BaselineDroppedRead(name=d.name, group=d.group, reason=d.reason, detail=d.detail)
                for d in result.dropped
            ],
            seconds=result.seconds,
            notes=NOTES,
        )
