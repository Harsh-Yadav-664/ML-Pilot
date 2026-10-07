"""Task specs per project: validate against a connection's schema, version, confirm (#49).

Rules:
* A draft is edited in place.
* A confirmed spec is never changed. Editing it saves a new version (a draft with the next
  version number) and leaves the old row, and every run that points at it, as it was.
* Confirming needs a spec with no errors; warnings do not block.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DataVersion, Run
from app.db.models.task_spec import TaskSpec as TaskSpecRow
from app.schemas.tasks import (
    SpecIssueRead,
    TaskSpecInput,
    TaskSpecRead,
    TaskSpecValidation,
)
from app.services.connection_service import ConnectionService
from ml.data.schema_graph import SchemaGraph
from ml.data.snapshot import DataDescription
from ml.tasks.spec import (
    SpecError,
    SpecIssue,
    TaskSpec,
    from_yaml,
    schema_fingerprint,
    to_yaml,
    validate_against,
)

CONFIRMED_BY = "local user"


def _read_issue(i: SpecIssue) -> SpecIssueRead:
    return SpecIssueRead(path=i.path, message=i.message, severity=i.severity)


def _utc(ts: datetime | None) -> datetime | None:
    """SQLite hands back times without a zone; they are stored in UTC."""
    return ts if ts is None or ts.tzinfo else ts.replace(tzinfo=UTC)


def parse(text: str) -> TaskSpec:
    """The spec, or HTTP 422 with every field problem."""
    try:
        return from_yaml(text)
    except SpecError as e:
        raise HTTPException(422, detail={"issues": [i.as_dict() for i in e.issues]}) from None


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
            await self.db.flush()
            return await self._read(row, issues)
        newest = await self.db.scalar(
            select(func.max(TaskSpecRow.version)).where(
                TaskSpecRow.project_id == project_id, TaskSpecRow.name == row.name
            )
        )
        new = self._new_row(project_id, spec, data.connection_id, (newest or row.version) + 1)
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
            used_by_runs=used or 0,
            created_at=_utc(row.created_at) or datetime.now(UTC),
            issues=[_read_issue(i) for i in issues],
        )
