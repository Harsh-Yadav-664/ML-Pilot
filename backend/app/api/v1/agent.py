"""The auto-optimize agent of a project: start a job and poll it."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ai.gateway import AIGateway
from app.api.deps import DBSession, ProjectID
from app.core.config import settings
from app.schemas.api import AutoOptimizeRequest, JobStatus
from app.services import data_service
from ml.agents.decision_agent import DecisionAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/agent", tags=["agent"])

# In-memory job table; the persistent job runner replaces it (#91).
JOBS: dict[str, JobStatus] = {}


async def _run_agent_task(
    job_id: str, project_id: str, dataset_path: str, target_column: str, n_hypotheses: int
) -> None:
    agent = DecisionAgent(AIGateway(settings), settings)
    try:
        result = await agent.run_optimization_loop(
            dataset_path, target_column, n_hypotheses, project_id=project_id
        )
        JOBS[job_id] = JobStatus(job_id=job_id, status="completed", result=result)
    except Exception as e:  # recorded on the job as status 'failed' with the message
        logger.exception("Auto-optimize failed")
        JOBS[job_id] = JobStatus(job_id=job_id, status="failed", error=str(e))


@router.post("/auto-optimize", response_model=JobStatus)
async def start_auto_optimize(
    request: AutoOptimizeRequest,
    project_id: ProjectID,
    db: DBSession,
    background_tasks: BackgroundTasks,
) -> JobStatus:
    """Start the autonomous optimization loop on one data version."""
    path = await data_service.version_path(db, request.data_version_id)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(job_id=job_id, status="running")
    background_tasks.add_task(
        _run_agent_task, job_id, project_id, str(path), request.target_column, request.n_hypotheses
    )
    return JOBS[job_id]


@router.get("/auto-optimize/{job_id}", response_model=JobStatus)
async def get_auto_optimize(project_id: ProjectID, job_id: str) -> JobStatus:
    """Poll an auto-optimize job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
