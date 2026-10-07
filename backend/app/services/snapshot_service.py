"""Take a snapshot of a connected database, or record its live state, as a data version (#97)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import DataVersion
from app.schemas.snapshot import DbVersionRead, SnapshotRequest
from app.services.connection_service import ConnectionService
from app.services.data_version_service import database_version_read, record_database_version
from ml.data.engine import EngineError
from ml.data.snapshot import SnapshotLimits, create_snapshot, live_version, now_utc
from ml.data.sources import ConnectionFailure
from ml.data.workspace import workspace_for

AsyncProgress = Callable[[str, float], Coroutine[Any, Any, None]]
CLOCK_SKEW = timedelta(minutes=2)


def resolve_as_of(as_of: datetime | None) -> datetime:
    """The cutoff to use: now, or what the caller asked (UTC); never in the future."""
    now = now_utc()
    if as_of is None:
        return now
    aware = as_of.astimezone(UTC) if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    if aware > now + CLOCK_SKEW:
        raise HTTPException(422, f"as_of {aware:%Y-%m-%d %H:%M:%S} UTC is in the future")
    return aware


class SnapshotService:
    def __init__(self, db: AsyncSession, limits: SnapshotLimits | None = None) -> None:
        self.db = db
        self.limits = limits or SnapshotLimits()

    async def take(
        self,
        project_id: str,
        connection_id: str,
        request: SnapshotRequest,
        progress: AsyncProgress | None = None,
    ) -> DbVersionRead:
        """Read the database as asked and record the data version (once per distinct data)."""
        connections = ConnectionService(self.db)
        conn = await connections.get(project_id, connection_id)
        graph = await connections.schema_graph(conn)
        source = connections.source(conn)
        as_of = resolve_as_of(request.as_of)
        who = {"id": conn.id, "name": conn.name}
        loop = asyncio.get_running_loop()

        def report(fraction: float, message: str) -> None:
            if progress is not None:
                asyncio.run_coroutine_threadsafe(progress(message, fraction), loop).result(30)

        try:
            if request.mode == "live":
                info = await asyncio.to_thread(
                    live_version,
                    source,
                    graph,
                    tables=request.tables,
                    as_of=as_of,
                    limits=self.limits,
                    connection=who,
                    scope=project_id,
                )
            else:
                workspace = workspace_for(project_id, datasets.PROJECTS_DIR)
                info = await asyncio.to_thread(
                    create_snapshot,
                    source,
                    graph,
                    workspace,
                    tables=request.tables,
                    columns=request.columns,
                    as_of=as_of,
                    limits=self.limits,
                    connection=who,
                    progress=report,
                    scope=project_id,
                )
        except ConnectionFailure as e:
            raise HTTPException(502, f"Could not read the database: {e}") from None
        except EngineError as e:
            raise HTTPException(422, str(e)) from None
        row = await record_database_version(self.db, project_id, info)
        return database_version_read(row)

    async def list(self, project_id: str) -> list[DbVersionRead]:
        rows = await self.db.scalars(
            select(DataVersion)
            .where(
                DataVersion.project_id == project_id,
                DataVersion.kind.in_(["db_snapshot", "db_live"]),
            )
            .order_by(DataVersion.created_at.desc())
        )
        return [database_version_read(r) for r in rows.all()]

    async def get(self, project_id: str, version_id: str) -> DbVersionRead:
        row = await self.db.get(DataVersion, version_id)
        if (
            row is None
            or row.project_id != project_id
            or row.kind not in ("db_snapshot", "db_live")
        ):
            raise HTTPException(404, "Data version not found in this project")
        return database_version_read(row)
