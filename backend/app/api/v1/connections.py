"""Saved database connections (Postgres, SQLite, DuckDB). Passwords are write-only."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import DBSession, ProjectID
from app.schemas.connection import (
    ConnectionCreate,
    ConnectionRead,
    ConnectionTestResult,
    ConnectionUpdate,
    TableStatsRead,
)
from app.services.connection_service import ConnectionService, to_read
from ml.data.schema_graph import SchemaGraph, SchemaOverrides

router = APIRouter(prefix="/projects/{project_id}/connections", tags=["connections"])


@router.post("/", response_model=ConnectionRead, status_code=status.HTTP_201_CREATED)
async def create_connection(
    data: ConnectionCreate, project_id: ProjectID, db: DBSession
) -> ConnectionRead:
    return to_read(await ConnectionService(db).create(project_id, data))


@router.get("/", response_model=list[ConnectionRead])
async def list_connections(project_id: ProjectID, db: DBSession) -> list[ConnectionRead]:
    return [to_read(c) for c in await ConnectionService(db).list(project_id)]


@router.get("/{connection_id}", response_model=ConnectionRead)
async def get_connection(
    connection_id: str, project_id: ProjectID, db: DBSession
) -> ConnectionRead:
    return to_read(await ConnectionService(db).get(project_id, connection_id))


@router.patch("/{connection_id}", response_model=ConnectionRead)
async def update_connection(
    connection_id: str, data: ConnectionUpdate, project_id: ProjectID, db: DBSession
) -> ConnectionRead:
    return to_read(await ConnectionService(db).update(project_id, connection_id, data))


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(connection_id: str, project_id: ProjectID, db: DBSession) -> Response:
    await ConnectionService(db).delete(project_id, connection_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{connection_id}/test", response_model=ConnectionTestResult)
async def test_connection(
    connection_id: str, project_id: ProjectID, db: DBSession
) -> ConnectionTestResult:
    """Connect read-only, report the server version and what the role could write."""
    svc = ConnectionService(db)
    return await svc.test(await svc.get(project_id, connection_id))


@router.get("/{connection_id}/schema", response_model=SchemaGraph)
async def get_schema(connection_id: str, project_id: ProjectID, db: DBSession) -> SchemaGraph:
    """Tables, columns, keys (declared and inferred) and time columns, with the user's
    overrides applied. Read through the SQL guard, so it is read-only."""
    svc = ConnectionService(db)
    return await svc.schema_graph(await svc.get(project_id, connection_id))


@router.patch("/{connection_id}/schema", response_model=SchemaGraph)
async def update_schema(
    connection_id: str, data: SchemaOverrides, project_id: ProjectID, db: DBSession
) -> SchemaGraph:
    """Store overrides (time columns, static tables, edges to add or remove). A field that is
    given replaces the stored one; a field left out is unchanged. Unknown names are a 422."""
    svc = ConnectionService(db)
    return await svc.update_schema_overrides(await svc.get(project_id, connection_id), data)


@router.get("/{connection_id}/tables/{table}/stats", response_model=TableStatsRead)
async def get_table_stats(
    connection_id: str,
    table: str,
    project_id: ProjectID,
    db: DBSession,
    refresh: bool = False,
) -> TableStatsRead:
    """Per-column statistics of one table, computed by aggregate SQL in the database (no rows
    are pulled), sampled above a million rows and cached. ``table`` is the table's ``key`` from
    the schema graph. ``refresh=true`` recomputes."""
    svc = ConnectionService(db)
    return await svc.table_stats(await svc.get(project_id, connection_id), table, refresh=refresh)
