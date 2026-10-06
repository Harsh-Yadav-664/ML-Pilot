"""Datasets API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession
from app.schemas.common import PaginatedResponse
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetUpdate
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(data: DatasetCreate, db: DBSession) -> DatasetRead:
    svc = DatasetService(db)
    dataset = await svc.create(data)
    return DatasetRead.model_validate(dataset)


@router.get("/project/{project_id}", response_model=PaginatedResponse[DatasetRead])
async def list_datasets(
    project_id: str,
    db: DBSession,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedResponse[DatasetRead]:
    svc = DatasetService(db)
    datasets, total = await svc.list_by_project(project_id, page, page_size)
    return PaginatedResponse(
        items=[DatasetRead.model_validate(d) for d in datasets],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
        has_prev=page > 1,
    )


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: str, db: DBSession) -> DatasetRead:
    svc = DatasetService(db)
    dataset = await svc.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetRead.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(dataset_id: str, data: DatasetUpdate, db: DBSession) -> DatasetRead:
    svc = DatasetService(db)
    dataset = await svc.update(dataset_id, data)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetRead.model_validate(dataset)
