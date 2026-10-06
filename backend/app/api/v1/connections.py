"""Saved database connections (Postgres, SQLite, DuckDB). Passwords are write-only."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import DBSession, ProjectID
from app.schemas.connection import (
    ConnectionCreate,
    ConnectionRead,
    ConnectionTestResult,
    ConnectionUpdate,
)
from app.services.connection_service import ConnectionService, to_read

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
