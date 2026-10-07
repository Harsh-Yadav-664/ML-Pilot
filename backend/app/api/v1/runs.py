"""Relational runs (#58): start the loop, read where a run stands, list its features."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DBSession, ProjectID
from app.schemas.runs import RunFeaturesRead, RunLoopRequest, RunLoopStarted, RunStateRead
from app.services.run_service import RunService

router = APIRouter(prefix="/projects/{project_id}/runs", tags=["runs"])


@router.post("/{run_id}/start", response_model=RunLoopStarted, status_code=202)
async def start_run(
    run_id: str, project_id: ProjectID, db: DBSession, body: RunLoopRequest | None = None
) -> RunLoopStarted:
    """Start the run as a background job: the baseline, then one proposed feature per round,
    each kept only if its paired gain on time-ordered folds of the training rows beats the
    margin, until a budget is reached, the rounds are used up or no feature has helped for
    ``patience`` rounds. The test rows are scored once, at the end. Needs a run made with
    ``POST .../tasks/{task}/runs`` on a snapshot. Cancel through the job."""
    return await RunService(db).start(project_id, run_id, body or RunLoopRequest())


@router.get("/{run_id}", response_model=RunStateRead)
async def get_run(run_id: str, project_id: ProjectID, db: DBSession) -> RunStateRead:
    return await RunService(db).state(project_id, run_id)


@router.get("/{run_id}/features", response_model=RunFeaturesRead)
async def get_run_features(run_id: str, project_id: ProjectID, db: DBSession) -> RunFeaturesRead:
    """Every feature of the run: the baseline's, and each proposal with its stage or gain."""
    return await RunService(db).features(project_id, run_id)
