"""A relational run (#58): start it as a job, run the loop, record every step.

``start`` checks the run and queues the job. ``execute`` is what the job does: build the
baseline on the run's snapshot, then run the loop of ``ml.agents.run_loop``. Everything is
recorded as it happens, in short transactions of its own, so a run that is cancelled, stopped
or interrupted always has its last champion on record:

* the baseline is the first champion (an Experiment, as in ``baseline_service``);
* each proposal is a Feature row with the stage that stopped it, or its paired gain;
* each accepted feature is a new Experiment (the new champion, parent = the previous one);
* at the end, the champion refit on train + validation is an Experiment holding the test
  metrics, scored once. A cancelled run has no test metrics.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from app.core import datasets
from app.db.models import Experiment, Feature, Run
from app.jobs import runner
from app.jobs.runner import JobCancelled, JobContext
from app.schemas.domain import FeatureRead
from app.schemas.runs import RunFeaturesRead, RunLoopRequest, RunLoopStarted, RunStateRead
from app.services.baseline_service import BaselineService, load_world
from app.services.connection_service import ConnectionService
from app.services.feature_proposals import feature_row
from app.services.privacy_service import PrivacyService
from app.services.task_service import TaskService
from ml.agents import run_loop as loop
from ml.data.engine import EngineError
from ml.data.profiling.db_stats import TableStats
from ml.data.workspace import workspace_for
from ml.features.baseline import BaselineError, build_baseline, flatten_graph, typed_times
from ml.features.engine import Budget, FeatureEngine
from ml.features.llm_sql import FeatureProposer
from ml.tasks import labels
from ml.validation.splits import SplitError, TemporalSplitPlan

KIND = "relational_run"
MAX_TABLES_PROFILED = 40

NOTES = [
    (
        "The language model proposes features; whether one is kept is decided in code, by its "
        "paired gain on time-ordered folds of the training rows."
    ),
    (
        "The test rows were scored once, after the last round, by a model refit on train and "
        "validation. They played no part in any decision."
    ),
]


def _aware(ts: datetime | None) -> datetime | None:
    if ts is None or ts.tzinfo is not None:
        return ts
    return ts.replace(tzinfo=UTC)


class RunService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _run(self, project_id: str, run_id: str) -> Run:
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id or run.task_spec_id is None:
            raise HTTPException(404, "Run not found in this project")
        return run

    async def start(self, project_id: str, run_id: str, body: RunLoopRequest) -> RunLoopStarted:
        run = await self._run(project_id, run_id)
        if run.status != "created" or run.champion_experiment_id is not None:
            raise HTTPException(409, "This run was already started; start a new run")
        if run.data_version_id is None:
            raise HTTPException(
                422, "The run was started on the live database; the loop needs a snapshot"
            )
        job = await runner.enqueue(
            self.db,
            project_id=project_id,
            kind=KIND,
            params={"run_id": run_id, "request": body.model_dump()},
        )
        job.run_id = run_id
        run.status = "queued"
        await self.db.flush()
        return RunLoopStarted(run_id=run_id, job_id=job.id)

    async def state(self, project_id: str, run_id: str) -> RunStateRead:
        run = await self._run(project_id, run_id)
        rows = (await self.db.scalars(select(Feature).where(Feature.run_id == run_id))).all()
        proposals = [f for f in rows if f.kind == "llm_sql"]
        champion = (
            await self.db.get(Experiment, run.champion_experiment_id)
            if run.champion_experiment_id
            else None
        )
        info = run.manifest.get("run_loop", {}) if run.manifest else {}
        return RunStateRead(
            id=run.id,
            task_id=run.task_spec_id,
            data_version_id=run.data_version_id,
            status=run.status,  # type: ignore[arg-type]
            stop_reason=info.get("stop_reason"),
            budget=run.budget or {},
            budget_used=run.budget_used or {},
            champion_experiment_id=run.champion_experiment_id,
            champion_val_metrics=champion.val_metrics if champion else None,
            test_metrics=champion.test_metrics if champion else None,
            test_error=info.get("test_error"),
            rounds=len(proposals),
            accepted=sum(f.status == "accepted" for f in proposals),
            error=run.error,
            notes=NOTES,
            started_at=_aware(run.started_at),
            finished_at=_aware(run.finished_at),
        )

    async def features(self, project_id: str, run_id: str) -> RunFeaturesRead:
        await self._run(project_id, run_id)
        rows = await self.db.scalars(
            select(Feature).where(Feature.run_id == run_id).order_by(Feature.created_at)
        )
        return RunFeaturesRead(
            run_id=run_id, features=[FeatureRead.model_validate(f) for f in rows.all()]
        )


class _Sink:
    """Writes each round as it ends: the proposal, and for an accepted one the new champion."""

    def __init__(self, run_id: str, previous_experiment: str, task_name: str, seed: int) -> None:
        self.run_id = run_id
        self.parent = previous_experiment
        self.task_name = task_name
        self.seed = seed
        self.position = 0

    async def round_done(self, rnd: loop.Round, champion: loop.Champion) -> None:
        self.position += 1
        async with runner.sessions()() as db:
            run = await db.get(Run, self.run_id)
            assert run is not None
            db.add(feature_row(self.run_id, rnd.record, self.position, rnd.gain))
            if rnd.accepted and rnd.gain is not None and rnd.record.proposal is not None:
                exp = Experiment(
                    id=str(uuid.uuid4()),
                    parent_id=self.parent,
                    project_id=run.project_id,
                    dataset_version=str(run.data_version_id)[:50],
                    hypothesis=rnd.record.proposal.rationale,
                    change_description=f"Added feature {rnd.record.proposal.name}",
                    model_name="LGBMClassifier",
                    feature_set=champion.names,
                    metrics=champion.metrics,
                    val_metrics=champion.metrics,
                    status="completed",
                    decision="keep",
                    decision_reason=(
                        f"Paired gain {rnd.gain.mean_gain:+.4f} PR-AUC on the temporal folds "
                        f"beats the margin {rnd.gain.margin:.4f}"
                    ),
                    decision_mode="rule",
                    decision_detail=rnd.gain.to_dict(),
                    runtime_seconds=rnd.seconds,
                    run_id=self.run_id,
                    data_version_id=run.data_version_id,
                    manifest={"kind": "champion", "round": rnd.number, "seed": self.seed},
                )
                db.add(exp)
                await db.flush()
                run.champion_experiment_id = exp.id
                self.parent = exp.id
            await db.commit()


def _budget(request: RunLoopRequest) -> Budget:
    return Budget(request.max_cost_usd, request.max_seconds, request.max_proposals)


async def _fail(run_id: str, message: str) -> None:
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        if run is not None:
            run.status, run.error, run.finished_at = "failed", message[:2000], datetime.now(UTC)
            await db.commit()


async def execute(
    ctx: JobContext, params: dict[str, Any], make_gateway: Any = None
) -> dict[str, Any]:
    """The job: baseline, loop, final model. Raises (and records ``failed``) on an error."""
    run_id = params["run_id"]
    request = RunLoopRequest.model_validate(params["request"])
    gateway: AIGateway = (make_gateway or _default_gateway)()
    try:
        return await _execute(ctx, run_id, request, gateway)
    except JobCancelled:
        await _cancelled(run_id)
        raise
    except (BaselineError, SplitError, labels.LabelError, EngineError, HTTPException) as e:
        await _fail(run_id, str(getattr(e, "detail", e)))
        raise RuntimeError(str(getattr(e, "detail", e))) from e
    except Exception as e:
        await _fail(run_id, f"{type(e).__name__}: {e}")
        raise


def _default_gateway() -> AIGateway:
    from app.jobs.handlers import make_gateway

    return make_gateway()


async def _cancelled(run_id: str) -> None:
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        if run is not None:
            run.status, run.finished_at = "cancelled", datetime.now(UTC)
            run.error = "cancelled by request; the champion recorded so far is kept"
            await db.commit()


async def _execute(
    ctx: JobContext, run_id: str, request: RunLoopRequest, gateway: AIGateway
) -> dict[str, Any]:
    await ctx.step("Reading the snapshot and building the baseline", 0.02)
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        if run is None or run.task_spec_id is None or run.data_version_id is None:
            raise HTTPException(404, "The run disappeared or has no snapshot")
        project_id, task_id, version_id = run.project_id, run.task_spec_id, run.data_version_id
        plan = TemporalSplitPlan.model_validate(run.split_plan)
        seed = run.seed
        built = await TaskService(db).labels_for_run(project_id, task_id, version_id)
        if built.mode != "snapshot" or built.row is None or built.row.connection_id is None:
            raise HTTPException(422, "The loop needs a snapshot data version of a saved task")
        connections = ConnectionService(db)
        conn = await connections.get(project_id, built.row.connection_id)
        graph = await connections.schema_graph(conn)
        stats: dict[str, TableStats] = {}
        for table in graph.tables[:MAX_TABLES_PROFILED]:
            stats[table.key] = (await connections.table_stats(conn, table.key)).stats
        builder = await PrivacyService(db).builder(project_id)
        run.status, run.started_at = "running", datetime.now(UTC)
        run.budget = _budget(request).as_dict()
        await db.commit()  # the prompt log writes on its own connection: hold no write
    spec = built.spec
    workspace = workspace_for(project_id, datasets.PROJECTS_DIR)
    world = await asyncio.to_thread(load_world, workspace, graph, version_id, task_id)
    baseline = await asyncio.to_thread(
        build_baseline, spec, world.graph, world.tables, world.labels, plan, seed=seed
    )
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        await BaselineService(db)._record(run, spec.name, baseline, world.absent)
        run.status = "running"
        await db.commit()
        previous = str(run.champion_experiment_id)
    await ctx.emit(
        "baseline",
        val_pr_auc=baseline.metrics["pr_auc"],
        base_rate=baseline.metrics["base_rate"],
        features=len(baseline.features),
    )
    cache = datasets.PROJECTS_DIR / project_id / "feature_cache"
    # the proposer flattens the graph and types the times; the engine has to run on the same
    flat, _ = flatten_graph(world.graph, spec.entity.table)
    engine = FeatureEngine(
        flat,
        typed_times(world.tables, flat),
        baseline.labels,  # type: ignore[arg-type]
        data_version_id=version_id,
        label_version=labels.label_table_ref(task_id, version_id).name,
        cache_dir=cache,
        timeout_s=request.feature_timeout_seconds,
        seed=seed,
    )
    proposer = FeatureProposer(
        spec=spec,
        graph=world.graph,
        tables=world.tables,
        baseline=baseline,
        gateway=gateway,
        builder=builder,
        stats=stats,
        engine=engine,
    )
    config = loop.RunConfig(
        max_rounds=request.max_rounds,
        patience=request.patience,
        budget=_budget(request),
        seed=seed,
    )
    state, scorer = await asyncio.to_thread(loop.start_state, baseline, config)
    sink = _Sink(run_id, previous, spec.name, seed)
    try:
        outcome = await loop.run_loop(
            baseline, proposer, config, hooks=ctx, sink=sink, state=state, scorer=scorer
        )
    finally:
        engine.close()
    await _finish(run_id, outcome, sink.parent, spec.name, engine, seed)
    return {
        "run_id": run_id,
        "status": outcome.status,
        "stop_reason": outcome.stop_reason,
        "accepted": len(outcome.accepted),
        "rounds": len(outcome.rounds),
    }


async def _finish(
    run_id: str,
    outcome: loop.RunOutcome,
    parent: str,
    task_name: str,
    engine: FeatureEngine,
    seed: int,
) -> None:
    now = datetime.now(UTC)
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        final = Experiment(
            id=str(uuid.uuid4()),
            parent_id=parent,
            project_id=run.project_id,
            dataset_version=str(run.data_version_id)[:50],
            hypothesis="Final model: the champion refit on train and validation, test scored once",
            change_description=f"{len(outcome.champion.names)} features; stopped: {outcome.stop_reason}",
            model_name="LGBMClassifier",
            feature_set=outcome.champion.names,
            metrics=outcome.test,  # None if the test rows could not be scored
            val_metrics=outcome.champion.metrics,
            test_metrics=outcome.test,
            status="completed",
            decision="keep",
            decision_reason="The champion at the end of the run",
            decision_mode="rule",
            run_id=run_id,
            data_version_id=run.data_version_id,
            manifest={
                "kind": "final",
                "task": task_name,
                "seed": seed,
                "stop_reason": outcome.stop_reason,
                "test_error": outcome.test_error,
            },
        )
        db.add(final)
        await db.flush()
        run.champion_experiment_id = final.id
        run.status = outcome.status
        run.finished_at = now
        run.budget_used = {**outcome.used, "stopped": outcome.stop_reason}
        run.manifest = {
            **run.manifest,
            "run_loop": {
                "stop_reason": outcome.stop_reason,
                "rounds": len(outcome.rounds),
                "accepted": [r.record.proposal.name for r in outcome.accepted if r.record.proposal],
                "test_error": outcome.test_error,
                "feature_engine": {**engine.counts, "sampling": engine.sampling.as_dict()},
                "at": now.isoformat(),
            },
        }
        await db.commit()
