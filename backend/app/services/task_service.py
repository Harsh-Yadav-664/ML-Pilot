"""Task specs per project: validate against a connection's schema, version, confirm (#49).

Rules:
* A draft is edited in place.
* A confirmed spec is never changed. Editing it saves a new version (a draft with the next
  version number) and leaves the old row, and every run that points at it, as it was.
* Confirming needs a spec with no errors; warnings do not block.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from app.core import datasets
from app.db.models import DataVersion, Run
from app.db.models.task_spec import TaskSpec as TaskSpecRow
from app.schemas.tasks import (
    CutoffCountRead,
    DroppedCutoffRead,
    FeasibilityRead,
    FoldRead,
    LabelPreview,
    LabelPreviewRequest,
    RunStartRequest,
    SpecIssueRead,
    SplitPreview,
    SplitPreviewRequest,
    TaskDraft,
    TaskDraftRequest,
    TaskRunRead,
    TaskSpecInput,
    TaskSpecRead,
    TaskSpecValidation,
    TimelineEntry,
)
from app.services.connection_service import ConnectionService
from app.services.privacy_service import PrivacyService
from ml.data.engine import EngineError
from ml.data.profiling.db_stats import TableStats
from ml.data.schema_graph import SchemaGraph
from ml.data.snapshot import DataDescription
from ml.data.sources import ConnectionFailure
from ml.data.workspace import workspace_for
from ml.tasks import labels, nl_to_spec
from ml.tasks.feasibility import FeasibilityReport, Thresholds, check_feasibility
from ml.tasks.spec import (
    SpecError,
    SpecIssue,
    TaskSpec,
    from_yaml,
    schema_fingerprint,
    to_yaml,
    validate_against,
)
from ml.validation.splits import (
    SplitError,
    TemporalSplitPlan,
    expanding_folds,
    partition_cutoffs,
    timeline,
)

CONFIRMED_BY = "local user"
MAX_TABLES_PROFILED = 40  # tables whose statistics go into the drafting prompt


def _read_issue(i: SpecIssue) -> SpecIssueRead:
    return SpecIssueRead(path=i.path, message=i.message, severity=i.severity)


def _utc_time(value: Any) -> datetime:
    """Any cutoff or window end as an aware UTC datetime (the split code works on naive UTC)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)
    return ts.to_pydatetime()


def _utc(ts: datetime | None) -> datetime | None:
    """SQLite hands back times without a zone; they are stored in UTC."""
    return ts if ts is None or ts.tzinfo else ts.replace(tzinfo=UTC)


def parse(text: str) -> TaskSpec:
    """The spec, or HTTP 422 with every field problem."""
    try:
        return from_yaml(text)
    except SpecError as e:
        raise HTTPException(422, detail={"issues": [i.as_dict() for i in e.issues]}) from None


@dataclass(frozen=True)
class _BuiltLabels:
    row: TaskSpecRow
    spec: TaskSpec
    run: labels.LabelRun
    mode: str
    as_of: datetime
    version_id: str | None


class TaskService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # -- checking -----------------------------------------------------------------------------
    async def _graph(self, project_id: str, connection_id: str) -> SchemaGraph:
        connections = ConnectionService(self.db)
        return await connections.schema_graph(await connections.get(project_id, connection_id))

    async def _as_of(self, project_id: str, data_version_id: str | None) -> datetime | None:
        if data_version_id is None:
            return None
        row = await self.db.get(DataVersion, data_version_id)
        if (
            row is None
            or row.project_id != project_id
            or row.kind not in ("db_snapshot", "db_live")
        ):
            raise HTTPException(404, "Database data version not found in this project")
        return DataDescription.model_validate(row.source).as_of

    async def check(
        self, project_id: str, data: TaskSpecInput
    ) -> tuple[TaskSpec, SchemaGraph, list[SpecIssue]]:
        spec = parse(data.yaml)
        graph = await self._graph(project_id, data.connection_id)
        as_of = await self._as_of(project_id, data.data_version_id)
        return spec, graph, validate_against(spec, graph, as_of)

    async def validate(self, project_id: str, data: TaskSpecInput) -> TaskSpecValidation:
        try:
            spec = from_yaml(data.yaml)
        except SpecError as e:
            return TaskSpecValidation(issues=[_read_issue(i) for i in e.issues])
        _, graph, issues = await self.check(project_id, data)
        return TaskSpecValidation(
            spec=spec.model_dump(mode="json", exclude_none=True),
            issues=[_read_issue(i) for i in issues],
            schema_fingerprint=schema_fingerprint(graph),
        )

    async def draft(self, project_id: str, data: TaskDraftRequest, gateway: AIGateway) -> TaskDraft:
        """Draft a task spec from a question. Reads the schema and column statistics, asks the
        language model (or, offline, the rule-based drafter), validates, and saves nothing."""
        connections = ConnectionService(self.db)
        conn = await connections.get(project_id, data.connection_id)
        graph = await connections.schema_graph(conn)
        now = await self._as_of(project_id, data.data_version_id) or datetime.now(UTC)
        stats: dict[str, TableStats] = {}
        for table in graph.tables[:MAX_TABLES_PROFILED]:
            stats[table.key] = (await connections.table_stats(conn, table.key)).stats
        ends = nl_to_spec.data_end(graph, stats, now)
        # Profiling cached statistics (a write). The prompt log records the prompt on its own
        # connection, and SQLite allows one writer at a time: release this one first.
        await self.db.commit()
        builder = await PrivacyService(self.db).builder(project_id)
        try:
            draft = await nl_to_spec.draft_spec(
                data.question, graph, gateway=gateway, builder=builder, as_of=ends, stats=stats
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        source = {
            "question": draft.question,
            "decision_mode": draft.decision_mode,
            "assumptions": draft.assumptions,
            "repaired": draft.repaired,
            "llm": draft.llm,
            "drafted_at": datetime.now(UTC).isoformat(),
        }
        return TaskDraft(
            status=draft.status,
            question=draft.question,
            decision_mode=draft.decision_mode,
            clarifying_question=draft.clarifying_question,
            yaml=draft.yaml,
            spec=draft.spec.model_dump(mode="json", exclude_none=True) if draft.spec else None,
            description=nl_to_spec.describe_spec(draft.spec) if draft.spec else None,
            assumptions=draft.assumptions,
            issues=[_read_issue(i) for i in draft.issues],
            repaired=draft.repaired,
            data_ends=ends,
            source=source,
        )

    # -- storing ------------------------------------------------------------------------------
    async def create(self, project_id: str, data: TaskSpecInput) -> TaskSpecRead:
        spec, _, issues = await self.check(project_id, data)
        taken = await self.db.scalar(
            select(func.count())
            .select_from(TaskSpecRow)
            .where(TaskSpecRow.project_id == project_id, TaskSpecRow.name == spec.name)
        )
        if taken:
            raise HTTPException(
                409, f"A task named {spec.name!r} exists; edit it (PUT) to save a new version"
            )
        row = self._new_row(project_id, spec, data.connection_id, 1)
        row.draft_source = data.draft_source
        self.db.add(row)
        await self.db.flush()
        return await self._read(row, issues)

    async def update(self, project_id: str, task_id: str, data: TaskSpecInput) -> TaskSpecRead:
        row = await self.get_row(project_id, task_id)
        spec, _, issues = await self.check(project_id, data)
        if spec.name != row.name:
            raise HTTPException(
                422,
                detail={
                    "issues": [
                        {
                            "path": "name",
                            "message": f"a task keeps its name ({row.name!r}) across versions; "
                            "create a new task to use another name",
                            "severity": "error",
                        }
                    ]
                },
            )
        if row.status == "draft":
            row.yaml = to_yaml(spec)
            row.connection_id = data.connection_id
            if data.draft_source is not None:
                row.draft_source = data.draft_source
            await self.db.flush()
            return await self._read(row, issues)
        newest = await self.db.scalar(
            select(func.max(TaskSpecRow.version)).where(
                TaskSpecRow.project_id == project_id, TaskSpecRow.name == row.name
            )
        )
        new = self._new_row(project_id, spec, data.connection_id, (newest or row.version) + 1)
        new.draft_source = data.draft_source if data.draft_source is not None else row.draft_source
        self.db.add(new)
        await self.db.flush()
        return await self._read(new, issues)

    async def confirm(
        self, project_id: str, task_id: str, data_version_id: str | None
    ) -> TaskSpecRead:
        row = await self.get_row(project_id, task_id)
        if row.status == "confirmed":
            return await self._read(row, [])
        if row.connection_id is None:
            raise HTTPException(422, "The spec has no connection to check it against")
        _, graph, issues = await self.check(
            project_id,
            TaskSpecInput(
                yaml=row.yaml, connection_id=row.connection_id, data_version_id=data_version_id
            ),
        )
        if any(i.severity == "error" for i in issues):
            raise HTTPException(422, detail={"issues": [i.as_dict() for i in issues]})
        row.status = "confirmed"
        row.confirmed_by = CONFIRMED_BY
        row.confirmed_at = datetime.now(UTC)
        row.schema_fingerprint = schema_fingerprint(graph)
        await self.db.flush()
        return await self._read(row, issues)

    # -- labels -------------------------------------------------------------------------------
    async def _build_labels(
        self, project_id: str, task_id: str, request: LabelPreviewRequest
    ) -> _BuiltLabels:
        """Build the labels of a saved task (and read its coverage) on the data version asked for."""
        row = await self.get_row(project_id, task_id)
        if row.connection_id is None:
            raise HTTPException(422, "The task has no connection to read the data from")
        spec = from_yaml(row.yaml)
        connections = ConnectionService(self.db)
        conn = await connections.get(project_id, row.connection_id)
        graph = await connections.schema_graph(conn)
        description: DataDescription | None = None
        version_id = request.data_version_id
        if version_id is not None:
            version = await self.db.get(DataVersion, version_id)
            if (
                version is None
                or version.project_id != project_id
                or version.kind not in ("db_snapshot", "db_live")
            ):
                raise HTTPException(404, "Database data version not found in this project")
            description = DataDescription.model_validate(version.source)
        mode = "snapshot" if description is not None and description.mode == "snapshot" else "live"
        as_of = description.as_of if description else datetime.now(UTC)
        try:
            if description is not None and mode == "snapshot":
                assert version_id is not None
                workspace = workspace_for(project_id, datasets.PROJECTS_DIR)
                run = await asyncio.to_thread(
                    labels.run_on_snapshot,
                    spec,
                    graph,
                    workspace,
                    version_id,
                    description,
                    task_id=task_id,
                    materialize=request.materialize,
                )
            else:
                run = await asyncio.to_thread(
                    labels.run_on_source, spec, graph, connections.source(conn), as_of
                )
        except labels.LabelError as e:
            raise HTTPException(422, str(e)) from None
        except ConnectionFailure as e:
            raise HTTPException(502, f"Could not read the database: {e}") from None
        except EngineError as e:
            raise HTTPException(422, str(e)) from None
        return _BuiltLabels(row, spec, run, mode, as_of, version_id)

    async def preview_labels(
        self, project_id: str, task_id: str, request: LabelPreviewRequest
    ) -> LabelPreview:
        """Build the labels for a saved task and report the SQL, the counts per cutoff and the
        feasibility checks."""
        built = await self._build_labels(project_id, task_id, request)
        report = self._feasibility(built, request.thresholds)
        return self._label_preview(built.run, built.mode, built.as_of, report)

    @staticmethod
    def _feasibility(built: _BuiltLabels, thresholds: Thresholds | None) -> FeasibilityReport:
        return check_feasibility(
            built.spec,
            built.run.counts,
            dropped_cutoffs=len(built.run.compiled.dropped),
            coverage=built.run.coverage,
            thresholds=thresholds,
        )

    async def start_run(
        self, project_id: str, task_id: str, request: RunStartRequest
    ) -> TaskRunRead:
        """Record a run of a confirmed task, unless the feasibility checks block it.

        A blocked task is a 422 with the reasons. With ``override`` the run is recorded and its
        manifest keeps the report and the override. Training itself is not started here.
        """
        row = await self.get_row(project_id, task_id)
        if row.status != "confirmed":
            raise HTTPException(409, "Only a confirmed task can be run: confirm the spec first")
        built = await self._build_labels(
            project_id, task_id, LabelPreviewRequest(data_version_id=request.data_version_id)
        )
        report = self._feasibility(built, request.thresholds)
        if report.blocked and not request.override:
            raise HTTPException(
                422,
                detail={
                    "message": "The task is blocked by the feasibility checks. Fix the task or "
                    "start the run with override=true",
                    "reasons": report.reasons,
                    "feasibility": report.as_dict(),
                },
            )
        try:
            plan = TemporalSplitPlan.from_spec(built.spec).model_dump(mode="json")
        except SplitError as e:
            raise HTTPException(422, str(e)) from None
        manifest: dict[str, Any] = {
            "task": {"id": row.id, "name": row.name, "version": row.version},
            "data_version_id": built.version_id,
            "as_of": built.as_of.isoformat(),
            "feasibility": report.as_dict(),
            "override": (
                {
                    "used": True,
                    "reason": request.override_reason,
                    "blocked_by": report.reasons,
                    "at": datetime.now(UTC).isoformat(),
                }
                if report.blocked
                else {"used": False}
            ),
        }
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            task_spec_id=row.id,
            data_version_id=built.version_id,
            split_plan=plan,
            manifest=manifest,
            engine="not_selected",
            seed=42,
            status="created",
        )
        self.db.add(run)
        await self.db.flush()
        await self.db.refresh(run)
        return TaskRunRead(
            id=run.id,
            project_id=project_id,
            task_id=row.id,
            data_version_id=run.data_version_id,
            status="created",
            split_plan=run.split_plan,
            manifest=run.manifest,
            created_at=_utc(run.created_at) or datetime.now(UTC),
        )

    async def preview_split(
        self, project_id: str, task_id: str, request: SplitPreviewRequest
    ) -> SplitPreview:
        """How the task's cutoffs divide into train, validation and test (the run view's timeline)."""
        row = await self.get_row(project_id, task_id)
        spec = from_yaml(row.yaml)
        preview = await self.preview_labels(project_id, task_id, request)
        rows = {
            (
                pd.Timestamp(c.cutoff).tz_convert(UTC).tz_localize(None),
                pd.Timestamp(c.window_end).tz_convert(UTC).tz_localize(None),
            ): c.eligible
            for c in preview.cutoffs
        }
        try:
            plan = TemporalSplitPlan.from_spec(spec, request.folds)
            parts = partition_cutoffs(list(rows), plan)
            train_pairs = [p for p, part in parts.items() if part == "train"]
            folds = expanding_folds(train_pairs, request.folds)
        except SplitError as e:
            raise HTTPException(422, str(e)) from None

        def total(part: str) -> int:
            return sum(rows[p] for p, x in parts.items() if x == part)

        n_train, n_val, n_test = total("train"), total("val"), total("test")
        if not (n_train and n_val and n_test):
            raise HTTPException(
                422,
                "Training, validation or test would have no rows: check split.val_from, "
                "split.test_from, the horizon and the cutoff range",
            )
        ends = {part: [p[1] for p, x in parts.items() if x == part] for part in ("train", "val")}
        return SplitPreview(
            val_from=_utc_time(plan.val_from),
            test_from=_utc_time(plan.test_from),
            timeline=[
                TimelineEntry(
                    cutoff=_utc_time(t["cutoff"]),
                    window_end=_utc_time(t["window_end"]),
                    part=t["part"],
                    rows=t["rows"],
                )
                for t in timeline(parts, rows)
            ],
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_purged_train=total("purged_train"),
            n_purged_val=total("purged_val"),
            max_train_window_end=_utc_time(max(ends["train"])),
            max_val_window_end=_utc_time(max(ends["val"])),
            folds=[
                FoldRead(
                    n_train=sum(rows[p] for p in f_train),
                    n_val=sum(rows[p] for p in f_val),
                    val_from=_utc_time(min(c for c, _ in f_val)),
                    val_to=_utc_time(max(c for c, _ in f_val)),
                )
                for f_train, f_val in folds
            ],
            dropped_cutoffs=preview.dropped_cutoffs,
            note="Counts are label rows per part. Per-entity overlap is in the split summary of a split made with entity ids.",
        )

    @staticmethod
    def _label_preview(
        run: labels.LabelRun, mode: str, as_of: datetime, report: FeasibilityReport
    ) -> LabelPreview:
        dropped = run.compiled.dropped
        return LabelPreview(
            sql=run.compiled.sql,
            dialect=run.dialect,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            as_of=as_of,
            cutoffs=[
                CutoffCountRead(
                    cutoff=c.cutoff,
                    window_end=c.window_end,
                    eligible=c.eligible,
                    positives=c.positives,
                    base_rate=c.base_rate,
                    mean_label=c.mean_label,
                )
                for c in run.counts
            ],
            dropped_cutoffs=[
                DroppedCutoffRead(
                    cutoff=w.cutoff,
                    window_end=w.window_end,
                    reason="its label window ends after the data does",
                )
                for w in dropped
            ],
            total_rows=run.total_rows,
            table=f"{run.table.schema}.{run.table.name}" if run.table else None,
            feasibility=FeasibilityRead.model_validate(report.as_dict()),
        )

    # -- reading ------------------------------------------------------------------------------
    async def get_row(self, project_id: str, task_id: str) -> TaskSpecRow:
        row = await self.db.get(TaskSpecRow, task_id)
        if row is None or row.project_id != project_id:
            raise HTTPException(404, "Task spec not found in this project")
        return row

    async def get(self, project_id: str, task_id: str) -> TaskSpecRead:
        return await self._read(await self.get_row(project_id, task_id), [])

    async def list_specs(self, project_id: str, latest_only: bool = False) -> list[TaskSpecRead]:
        rows = (
            await self.db.scalars(
                select(TaskSpecRow)
                .where(TaskSpecRow.project_id == project_id)
                .order_by(TaskSpecRow.name, TaskSpecRow.version.desc())
            )
        ).all()
        if latest_only:
            newest: dict[str, TaskSpecRow] = {}
            for r in rows:  # ordered by name, then newest version first
                newest.setdefault(r.name, r)
            rows = list(newest.values())
        return [await self._read(r, []) for r in rows]

    # -- internals ----------------------------------------------------------------------------
    @staticmethod
    def _new_row(project_id: str, spec: TaskSpec, connection_id: str, version: int) -> TaskSpecRow:
        return TaskSpecRow(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=spec.name,
            connection_id=connection_id,
            version=version,
            yaml=to_yaml(spec),
            status="draft",
        )

    async def _read(self, row: TaskSpecRow, issues: list[SpecIssue]) -> TaskSpecRead:
        used = await self.db.scalar(
            select(func.count()).select_from(Run).where(Run.task_spec_id == row.id)
        )
        spec: dict[str, Any] = from_yaml(row.yaml).model_dump(mode="json", exclude_none=True)
        return TaskSpecRead(
            id=row.id,
            project_id=row.project_id,
            name=row.name,
            version=row.version,
            status=row.status,  # type: ignore[arg-type]
            yaml=row.yaml,
            spec=spec,
            connection_id=row.connection_id,
            schema_fingerprint=row.schema_fingerprint,
            confirmed_by=row.confirmed_by,
            confirmed_at=_utc(row.confirmed_at),
            draft_source=row.draft_source,
            used_by_runs=used or 0,
            created_at=_utc(row.created_at) or datetime.now(UTC),
            issues=[_read_issue(i) for i in issues],
        )
