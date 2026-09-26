"""Experiments API endpoints."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession
from app.db.session import AsyncSessionLocal
from app.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate
from app.schemas.common import PaginatedResponse
from app.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])

async def background_runner(experiment_id: str):
    """Background task to run experiment safely with its own DB session."""
    async with AsyncSessionLocal() as session:
        try:
            svc = ExperimentService(session)
            await svc.run_experiment_background(experiment_id)
        except Exception as e:
            logger.error(f"Background task failed for exp {experiment_id}: {e}")


@router.post("/", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    data: ExperimentCreate, 
    db: DBSession,
    background_tasks: BackgroundTasks
) -> ExperimentRead:
    svc = ExperimentService(db)
    exp = await svc.create(data)
    
    # Queue the actual ML execution in the background
    background_tasks.add_task(background_runner, exp.id)
    
    return ExperimentRead.model_validate(exp)


@router.get("/project/{project_id}", response_model=PaginatedResponse[ExperimentRead])
async def list_experiments(
    project_id: str,
    db: DBSession,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedResponse[ExperimentRead]:
    svc = ExperimentService(db)
    exps, total = await svc.list_by_project(project_id, page, page_size)
    return PaginatedResponse(
        items=[ExperimentRead.model_validate(e) for e in exps],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
        has_prev=page > 1,
    )


@router.get("/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(experiment_id: str, db: DBSession) -> ExperimentRead:
    svc = ExperimentService(db)
    exp = await svc.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentRead.model_validate(exp)


@router.patch("/{experiment_id}", response_model=ExperimentRead)
async def update_experiment(experiment_id: str, data: ExperimentUpdate, db: DBSession) -> ExperimentRead:
    svc = ExperimentService(db)
    exp = await svc.update(experiment_id, data)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentRead.model_validate(exp)
