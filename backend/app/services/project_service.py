"""Project CRUD service."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate


class ProjectService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: ProjectCreate, owner_id: str) -> Project:
        project = Project(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            **data.model_dump(),
        )
        self.db.add(project)
        await self.db.flush()
        return project

    async def get(self, project_id: str) -> Optional[Project]:
        result = await self.db.execute(select(Project).where(Project.id == project_id))
        return result.scalar_one_or_none()

    async def list_by_owner(
        self, owner_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Project], int]:
        offset = (page - 1) * page_size
        count_q = await self.db.execute(
            select(func.count()).select_from(Project).where(Project.owner_id == owner_id)
        )
        total = count_q.scalar_one()
        result = await self.db.execute(
            select(Project)
            .where(Project.owner_id == owner_id)
            .offset(offset)
            .limit(page_size)
            .order_by(Project.created_at.desc())
        )
        return list(result.scalars().all()), total

    async def update(self, project_id: str, data: ProjectUpdate) -> Optional[Project]:
        project = await self.get(project_id)
        if not project:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(project, field, value)
        await self.db.flush()
        return project

    async def delete(self, project_id: str) -> bool:
        project = await self.get(project_id)
        if not project:
            return False
        await self.db.delete(project)
        return True
