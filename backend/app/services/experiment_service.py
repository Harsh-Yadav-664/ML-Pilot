"""Experiment CRUD service."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


class ExperimentService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: ExperimentCreate) -> Experiment:
        experiment = Experiment(id=str(uuid.uuid4()), **data.model_dump())
        self.db.add(experiment)
        await self.db.flush()
        return experiment

    async def get(self, experiment_id: str) -> Optional[Experiment]:
        result = await self.db.execute(select(Experiment).where(Experiment.id == experiment_id))
        return result.scalar_one_or_none()

    async def list_by_project(
        self, project_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Experiment], int]:
        offset = (page - 1) * page_size
        count_q = await self.db.execute(
            select(func.count()).select_from(Experiment).where(Experiment.project_id == project_id)
        )
        total = count_q.scalar_one()
        result = await self.db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id)
            .offset(offset)
            .limit(page_size)
            .order_by(Experiment.created_at.desc())
        )
        return list(result.scalars().all()), total

    async def update(self, experiment_id: str, data: ExperimentUpdate) -> Optional[Experiment]:
        exp = await self.get(experiment_id)
        if not exp:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(exp, field, value)
        await self.db.flush()
        return exp
