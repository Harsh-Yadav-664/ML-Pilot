"""FastAPI dependency injection utilities."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from app.db.session import get_db

# Type alias for injected DB session
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Placeholder owner_id until auth is wired
DEFAULT_OWNER_ID = "00000000-0000-0000-0000-000000000001"


def get_owner_id() -> str:
    """Return the current user id. Placeholder until JWT auth is implemented."""
    return DEFAULT_OWNER_ID


OwnerID = Annotated[str, Depends(get_owner_id)]


def get_gateway() -> AIGateway:
    """Return the LLM gateway built from settings, with the prompt log (overridable in tests)."""
    from app.services.llm_gateway import make_gateway

    return make_gateway()


Gateway = Annotated[AIGateway, Depends(get_gateway)]


async def get_project_id(project_id: str, db: DBSession) -> str:
    """The `{project_id}` path parameter, checked to name an existing project (404 otherwise)."""
    from app.services.project_service import ProjectService

    if await ProjectService(db).get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_id


ProjectID = Annotated[str, Depends(get_project_id)]
