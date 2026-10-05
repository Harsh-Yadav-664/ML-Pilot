"""Experiments API endpoints."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession, Gateway
from app.db.session import AsyncSessionLocal
from app.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate, ExperimentSuggestRequest, ExperimentSuggestResponse
from app.schemas.common import PaginatedResponse
from app.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("/suggest", response_model=ExperimentSuggestResponse)
async def suggest_experiments(
    data: ExperimentSuggestRequest, db: DBSession, gateway: Gateway
) -> ExperimentSuggestResponse:
    """Ask the AI Agent to propose new experiments/features based on the dataset."""
    svc = ExperimentService(db)
    try:
        hypotheses = await svc.suggest_experiments(
            dataset_version=data.dataset_version,
            target_column=data.target_column,
            objective=data.objective,
            max_hypotheses=data.max_hypotheses,
            gateway=gateway,
        )
        return ExperimentSuggestResponse(
            dataset_version=data.dataset_version,
            objective=data.objective,
            hypotheses=hypotheses
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"Suggest API failed: {e}")
        raise HTTPException(status_code=502, detail=str(e)) from e

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
    # Commit before scheduling: the background run reads the row from its own session.
    await db.commit()

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
