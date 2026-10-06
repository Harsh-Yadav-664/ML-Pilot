"""The auto-optimize agent of a project. It runs as a durable job (see /jobs)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DBSession, ProjectID
from app.jobs import runner
from app.schemas.api import AutoOptimizeRequest, JobRead
from app.services import data_service

router = APIRouter(prefix="/projects/{project_id}/agent", tags=["agent"])


@router.post("/auto-optimize", response_model=JobRead)
async def start_auto_optimize(
    request: AutoOptimizeRequest, project_id: ProjectID, db: DBSession
) -> JobRead:
    """Queue the optimization loop on one data version. Follow it at /jobs/{id}."""
    path = await data_service.version_path(db, request.data_version_id)
    job = await runner.enqueue(
        db,
        project_id=project_id,
        kind="auto_optimize",
        params={
            "project_id": project_id,
            "data_version_id": request.data_version_id,
            "dataset_path": str(path),
            "target_column": request.target_column,
            "n_hypotheses": request.n_hypotheses,
        },
    )
    await db.commit()
    runner.notify()
    return JobRead.model_validate(job)
