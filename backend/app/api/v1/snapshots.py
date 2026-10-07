"""Data versions of a connected database: snapshots, live records and their list (#97)."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import DBSession, ProjectID
from app.jobs import runner
from app.schemas.api import JobRead
from app.schemas.snapshot import DbVersionRead, SnapshotRequest, SnapshotResponse
from app.services.connection_service import ConnectionService
from app.services.snapshot_service import SnapshotService, resolve_as_of

router = APIRouter(prefix="/projects/{project_id}", tags=["data versions"])


@router.post(
    "/connections/{connection_id}/snapshots",
    response_model=SnapshotResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_snapshot(
    connection_id: str,
    data: SnapshotRequest,
    project_id: ProjectID,
    db: DBSession,
    response: Response,
) -> SnapshotResponse:
    """Version what a run will read from the database.

    ``mode: snapshot`` (default) copies the tables (only the columns asked for, only rows with an
    event time up to ``as_of``) into the project through read-only guarded SELECTs, as a job whose
    steps and progress are at ``/jobs/{id}``; its result holds ``data_version_id``. ``mode: live``
    copies nothing and returns, at once, the row counts and latest event times at ``as_of`` with
    ``reproducible: false``.
    """
    await ConnectionService(db).get(project_id, connection_id)  # 404 before anything is queued
    service = SnapshotService(db)
    if data.mode == "live":
        response.status_code = status.HTTP_201_CREATED
        return SnapshotResponse(version=await service.take(project_id, connection_id, data))
    # The cutoff is when the user asked, not when the job gets its turn.
    pinned = data.model_copy(update={"as_of": resolve_as_of(data.as_of)})
    job = await runner.enqueue(
        db,
        project_id=project_id,
        kind="snapshot",
        params={
            "connection_id": connection_id,
            "request": pinned.model_dump(mode="json"),
        },
    )
    await db.commit()
    runner.notify()
    return SnapshotResponse(job=JobRead.model_validate(job))


@router.get("/db-versions", response_model=list[DbVersionRead])
async def list_db_versions(project_id: ProjectID, db: DBSession) -> list[DbVersionRead]:
    """Snapshot and live versions of this project's databases, newest first."""
    return await SnapshotService(db).list(project_id)


@router.get("/db-versions/{version_id}", response_model=DbVersionRead)
async def get_db_version(version_id: str, project_id: ProjectID, db: DBSession) -> DbVersionRead:
    return await SnapshotService(db).get(project_id, version_id)
