"""Project CRUD service."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.project import Project
from app.db.models.user import User
from app.schemas.project import ProjectCreate, ProjectUpdate


class ProjectService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: ProjectCreate, owner_id: str) -> Project:
        await self._ensure_owner(owner_id)
        project = Project(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            **data.model_dump(),
        )
        self.db.add(project)
        await self.db.flush()
        return project

    async def _ensure_owner(self, owner_id: str) -> None:
        """Create the local user on first use (single-user installs have no sign-up yet).

        Two requests can get here at once, so the insert runs in a savepoint and a
        duplicate is treated as "someone else created it", not as a failure.
        """
        if await self.db.get(User, owner_id) is not None:
            return
        try:
            async with self.db.begin_nested():
                self.db.add(
                    User(id=owner_id, email=f"{owner_id}@local.mlpilot", display_name="Local user")
                )
        except IntegrityError:
            # Another request inserted the same user first; make sure it is really there.
            if await self.db.get(User, owner_id) is None:
                raise

    async def get(self, project_id: str) -> Project | None:
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

    async def update(self, project_id: str, data: ProjectUpdate) -> Project | None:
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
