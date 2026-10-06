"""A project's privacy settings and the prompt builder made from them."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ai.context_builder import (
    LEVEL_SUMMARY,
    SAMPLE_VALUES_WARNING,
    ContextBuilder,
    PrivacyLevel,
    PrivacyPolicy,
)
from app.db import session as db_session
from app.db.models.project import Project
from app.schemas.privacy import PrivacyRead, PrivacySettings


def _normalise(entries: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        if entry.count(".") > 1 or entry.startswith(".") or entry.endswith("."):
            raise HTTPException(422, f"Not a column name or table.column: {raw!r}")
        seen.setdefault(entry.lower(), entry)
    return sorted(seen.values(), key=str.lower)


def to_read(level: PrivacyLevel, never_send: list[str]) -> PrivacyRead:
    return PrivacyRead(
        level=level,
        never_send=never_send,
        level_summaries={k.value: v for k, v in LEVEL_SUMMARY.items()},
        warning=SAMPLE_VALUES_WARNING if level is PrivacyLevel.allow_sample_values else None,
    )


class PrivacyService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _project(self, project_id: str) -> Project:
        project = await self.db.get(Project, project_id)
        if project is None:
            raise HTTPException(404, "Project not found")
        return project

    async def policy(self, project_id: str) -> PrivacyPolicy:
        return PrivacyPolicy.from_settings((await self._project(project_id)).settings)

    async def builder(self, project_id: str) -> ContextBuilder:
        return ContextBuilder(await self.policy(project_id), project_id)

    async def read(self, project_id: str) -> PrivacyRead:
        policy = await self.policy(project_id)
        return to_read(policy.level, sorted(policy.never_send, key=str.lower))

    async def update(self, project_id: str, data: PrivacySettings) -> PrivacyRead:
        project = await self._project(project_id)
        never_send = _normalise(data.never_send)
        # A new dict, so SQLAlchemy sees the JSON column change.
        project.settings = {
            **(project.settings or {}),
            "privacy": {"level": data.level.value, "never_send": never_send},
        }
        await self.db.flush()
        return to_read(data.level, never_send)


async def builder_for(project_id: str, db: AsyncSession | None = None) -> ContextBuilder:
    """The prompt builder of a project, for code that has no database session at hand."""
    if db is not None:
        return await PrivacyService(db).builder(project_id)
    async with db_session.AsyncSessionLocal() as session:
        return await PrivacyService(session).builder(project_id)
