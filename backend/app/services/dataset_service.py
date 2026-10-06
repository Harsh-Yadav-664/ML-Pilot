"""Dataset CRUD service."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: DatasetCreate) -> Dataset:
        dataset = Dataset(id=str(uuid.uuid4()), **data.model_dump())
        self.db.add(dataset)
        await self.db.flush()
        return dataset

    async def get(self, dataset_id: str) -> Dataset | None:
        result = await self.db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalar_one_or_none()

    async def list_by_project(
        self, project_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Dataset], int]:
        offset = (page - 1) * page_size
        count_q = await self.db.execute(
            select(func.count()).select_from(Dataset).where(Dataset.project_id == project_id)
        )
        total = count_q.scalar_one()
        result = await self.db.execute(
            select(Dataset)
            .where(Dataset.project_id == project_id)
            .offset(offset)
            .limit(page_size)
            .order_by(Dataset.created_at.desc())
        )
        return list(result.scalars().all()), total

    async def update(self, dataset_id: str, data: DatasetUpdate) -> Dataset | None:
        dataset = await self.get(dataset_id)
        if not dataset:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(dataset, field, value)
        await self.db.flush()
        return dataset
