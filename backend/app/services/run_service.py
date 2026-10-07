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
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from app.core import datasets
from app.db.models import (
    Experiment,
    Feature,
    Job,
    JobEvent,
    Run,
    RunCheckpoint,
    RunSuggestion,
    TaskSpec,
)
from app.jobs import runner
from app.jobs.runner import JobCancelled, JobContext
from app.schemas.runs import (
    CheckpointDecision,
    CheckpointRead,
    NarrationItem,
    RunFeatureRead,
    RunFeaturesRead,
    RunLoopRequest,
    RunLoopStarted,
    RunNarration,
    RunStateRead,
    SettingChange,
    SettingsPreview,
    SuggestionCreate,
    SuggestionRead,
)
from app.services.baseline_service import BaselineService, load_world
from app.services.connection_service import ConnectionService
from app.services.feature_proposals import feature_description, feature_row
from app.services.privacy_service import PrivacyService
from app.services.task_service import TaskService
from ml.agents import narration, settings_nl
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
POLL_SECONDS = 0.3  # how often a waiting checkpoint looks for the answer
LIVE = ("queued", "running")  # a run in these states can still be steered

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
        run.budget = settings_of(body)  # steerable from now on (apply_settings)
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
        manifest = run.manifest or {}
        info = manifest.get("run_loop", {})
        job_id = await self.db.scalar(
            select(Job.id).where(Job.run_id == run_id).order_by(Job.created_at.desc()).limit(1)
        )
        spec = await self.db.get(TaskSpec, run.task_spec_id)
        draft = (spec.draft_source or {}) if spec is not None else {}
        as_of = manifest.get("as_of")
        return RunStateRead(
            id=run.id,
            task_id=run.task_spec_id,
            task_name=spec.name if spec is not None else None,
            question=draft.get("question"),
            data_version_id=run.data_version_id,
            as_of=datetime.fromisoformat(as_of) if as_of else None,
            split_plan=run.split_plan or {},
            feasibility=manifest.get("feasibility"),
            job_id=job_id,
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
        return RunFeaturesRead(run_id=run_id, features=[_feature_read(f) for f in rows.all()])


def _feature_read(f: Feature) -> RunFeatureRead:
    out = RunFeatureRead.model_validate(f)
    out.description = feature_description(f)
    return out


def _checkpoint_read(c: RunCheckpoint) -> CheckpointRead:
    return CheckpointRead(
        id=c.id,
        run_id=c.run_id,
        round=c.round,
        kind=c.kind,
        state=c.state,  # type: ignore[arg-type]
        recommended=c.recommended,  # type: ignore[arg-type]
        payload=c.payload,
        note=c.note,
        timeout_seconds=c.timeout_seconds,
        created_at=_aware(c.created_at),  # type: ignore[arg-type]
        decided_at=_aware(c.decided_at),
    )


def _suggestion_read(s: RunSuggestion) -> SuggestionRead:
    return SuggestionRead(
        id=s.id,
        run_id=s.run_id,
        text=s.text,
        state=s.state,  # type: ignore[arg-type]
        created_at=_aware(s.created_at),  # type: ignore[arg-type]
        used_in_round=s.used_in_round,
    )


class SteerService:
    """What a person does to a run while it goes on (#59)."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _live_run(self, project_id: str, run_id: str) -> Run:
        run = await RunService(self.db)._run(project_id, run_id)
        if run.status not in LIVE:
            raise HTTPException(
                409, f"The run is {run.status}; only a queued or running run can be steered"
            )
        return run

    async def checkpoints(self, project_id: str, run_id: str) -> list[CheckpointRead]:
        await RunService(self.db)._run(project_id, run_id)
        rows = await self.db.scalars(
            select(RunCheckpoint)
            .where(RunCheckpoint.run_id == run_id)
            .order_by(RunCheckpoint.created_at, RunCheckpoint.round)
        )
        return [_checkpoint_read(c) for c in rows.all()]

    async def answer(
        self, project_id: str, run_id: str, checkpoint_id: str, body: CheckpointDecision
    ) -> CheckpointRead:
        await RunService(self.db)._run(project_id, run_id)
        row = await self.db.get(RunCheckpoint, checkpoint_id)
        if row is None or row.run_id != run_id:
            raise HTTPException(404, "Checkpoint not found in this run")
        state = "approved" if body.decision == "approve" else "vetoed"
        # only a pending question can be answered, and only once: the update says so atomically
        result = await self.db.execute(
            update(RunCheckpoint)
            .where(RunCheckpoint.id == checkpoint_id, RunCheckpoint.state == "pending")
            .values(state=state, note=body.note, decided_at=datetime.now(UTC))
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise HTTPException(409, f"This question was already answered ({row.state})")
        await self.db.flush()
        await self.db.refresh(row)
        return _checkpoint_read(row)

    async def suggestions(self, project_id: str, run_id: str) -> list[SuggestionRead]:
        await RunService(self.db)._run(project_id, run_id)
        rows = await self.db.scalars(
            select(RunSuggestion)
            .where(RunSuggestion.run_id == run_id)
            .order_by(RunSuggestion.created_at)
        )
        return [_suggestion_read(s) for s in rows.all()]

    async def suggest(self, project_id: str, run_id: str, body: SuggestionCreate) -> SuggestionRead:
        await self._live_run(project_id, run_id)
        row = RunSuggestion(id=str(uuid.uuid4()), run_id=run_id, text=body.text.strip())
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return _suggestion_read(row)

    async def preview_settings(self, project_id: str, run_id: str, message: str) -> SettingsPreview:
        run = await RunService(self.db)._run(project_id, run_id)
        return _preview(message, run.budget or {}, applied=False)

    async def apply_settings(self, project_id: str, run_id: str, message: str) -> SettingsPreview:
        """Read the sentence again, now, and write what it changes into the run. The loop
        picks the new settings up before its next round."""
        run = await self._live_run(project_id, run_id)
        current = dict(run.budget or {})
        diff = settings_nl.parse(message, current)
        if diff.changes:
            run.budget = {**current, **{k: v["to"] for k, v in diff.changes.items()}}
            await self.db.flush()
        return _preview(message, current, applied=True, diff=diff)

    async def narration(self, project_id: str, run_id: str) -> RunNarration:
        await RunService(self.db)._run(project_id, run_id)
        rows = (
            await self.db.execute(
                select(JobEvent)
                .join(Job, Job.id == JobEvent.job_id)
                .where(Job.run_id == run_id)
                .order_by(JobEvent.ts, JobEvent.seq)
            )
        ).scalars()
        items: list[NarrationItem] = []
        for e in rows:
            text = narration.narrate(e.type, e.payload)
            if text is not None:
                items.append(
                    NarrationItem(
                        seq=e.seq,
                        type=e.type,
                        text=text,
                        at=_aware(e.ts),
                        payload=e.payload,  # type: ignore[arg-type]
                    )
                )
        return RunNarration(run_id=run_id, items=items)


def _preview(
    message: str,
    current: dict[str, Any],
    *,
    applied: bool,
    diff: settings_nl.SettingsDiff | None = None,
) -> SettingsPreview:
    diff = diff or settings_nl.parse(message, current)
    return SettingsPreview(
        changes=[
            SettingChange(setting=k, **{"from": v["from"], "to": v["to"]})
            for k, v in diff.changes.items()
        ],
        unrecognised=diff.unrecognised,
        summary=diff.summary(),
        applied=applied and bool(diff.changes),
    )


def settings_of(request: RunLoopRequest) -> dict[str, Any]:
    """Everything a person can change while the run goes on, as stored in ``Run.budget``."""
    return {
        **_budget(request).as_dict(),
        "max_rounds": request.max_rounds,
        "patience": request.patience,
        "approval_mode": request.approval_mode,
        "checkpoint_timeout_seconds": request.checkpoint_timeout_seconds,
        "feature_timeout_seconds": request.feature_timeout_seconds,
    }


class DbControl:
    """The loop's ``Control``, answered from the database. It emits the events itself: the job
    owns its event numbers."""

    def __init__(self, ctx: JobContext, run_id: str, config: loop.RunConfig) -> None:
        self.ctx = ctx
        self.run_id = run_id
        self.config = config
        self.checkpoint_timeout = 300.0

    async def settings(self) -> loop.RunConfig | None:
        async with runner.sessions()() as db:
            run = await db.get(Run, self.run_id)
            stored = dict(run.budget or {}) if run else {}
        new = replace(
            self.config,
            budget=Budget(
                stored.get("max_cost_usd"),
                stored.get("max_seconds"),
                stored.get("max_proposals"),
            ),
            max_rounds=int(stored.get("max_rounds", self.config.max_rounds)),
            patience=int(stored.get("patience", self.config.patience)),
            approval_mode=stored.get("approval_mode", self.config.approval_mode),
        )
        self.checkpoint_timeout = float(
            stored.get("checkpoint_timeout_seconds", self.checkpoint_timeout)
        )
        changes = _changes(self.config, new)
        self.config = new
        if changes:
            await self.ctx.emit("settings_changed", changes=changes)
        return new

    async def hints(self, number: int) -> list[str]:
        async with runner.sessions()() as db:
            rows = (
                await db.scalars(
                    select(RunSuggestion)
                    .where(RunSuggestion.run_id == self.run_id, RunSuggestion.state == "new")
                    .order_by(RunSuggestion.created_at)
                )
            ).all()
            for row in rows:
                row.state, row.used_in_round = "used", number
            await db.commit()
            texts = [r.text for r in rows]
        for text in texts:
            await self.ctx.emit("suggestion", round=number, text=text)
        return texts

    async def approve(self, number: int, summary: dict[str, Any]) -> loop.Answer:
        checkpoint_id = str(uuid.uuid4())
        timeout = self.checkpoint_timeout
        async with runner.sessions()() as db:
            db.add(
                RunCheckpoint(
                    id=checkpoint_id,
                    run_id=self.run_id,
                    round=number,
                    kind="approve_feature",
                    recommended="approve",
                    payload=summary,
                    timeout_seconds=timeout,
                )
            )
            await db.commit()
        await self.ctx.emit(
            "checkpoint",
            round=number,
            checkpoint_id=checkpoint_id,
            summary={"name": summary["name"], "description": summary["description"]},
            timeout_seconds=timeout,
            recommended="approve",
        )
        waited_from = time.monotonic()
        while True:
            await self.ctx.check_cancelled()
            async with runner.sessions()() as db:
                row = await db.get(RunCheckpoint, checkpoint_id)
                assert row is not None
                state = row.state
            if state in ("approved", "vetoed"):
                answer = loop.Answer("approve" if state == "approved" else "veto", "reply")
                break
            if time.monotonic() - waited_from >= timeout:
                async with runner.sessions()() as db:
                    # a reply that arrived in the last moment wins over the timeout
                    result = await db.execute(
                        update(RunCheckpoint)
                        .where(RunCheckpoint.id == checkpoint_id, RunCheckpoint.state == "pending")
                        .values(state="timeout", decided_at=datetime.now(UTC))
                    )
                    await db.commit()
                if result.rowcount:  # type: ignore[attr-defined]
                    answer = loop.Answer("approve", "timeout")
                    break
                continue
            await asyncio.sleep(POLL_SECONDS)
        await self.ctx.emit(
            "checkpoint_answered", round=number, decision=answer.decision, how=answer.how
        )
        return answer


def _changes(old: loop.RunConfig, new: loop.RunConfig) -> dict[str, dict[str, Any]]:
    pairs = {
        "max_cost_usd": (old.budget.max_cost_usd, new.budget.max_cost_usd),
        "max_seconds": (old.budget.max_seconds, new.budget.max_seconds),
        "max_proposals": (old.budget.max_proposals, new.budget.max_proposals),
        "max_rounds": (old.max_rounds, new.max_rounds),
        "patience": (old.patience, new.patience),
        "approval_mode": (old.approval_mode, new.approval_mode),
    }
    return {k: {"from": a, "to": b} for k, (a, b) in pairs.items() if a != b}


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
            db.add(feature_row(self.run_id, rnd.record, self.position, rnd.gain, rnd.vetoed))
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
        # what start() stored wins: a person may have changed it while the job waited
        run.budget = {**settings_of(request), **(run.budget or {})}
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
        approval_mode=request.approval_mode,
    )
    control = DbControl(ctx, run_id, config)
    control.checkpoint_timeout = request.checkpoint_timeout_seconds
    state, scorer = await asyncio.to_thread(loop.start_state, baseline, config)
    sink = _Sink(run_id, previous, spec.name, seed)
    try:
        outcome = await loop.run_loop(
            baseline,
            proposer,
            config,
            hooks=ctx,
            sink=sink,
            state=state,
            scorer=scorer,
            control=control,
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
