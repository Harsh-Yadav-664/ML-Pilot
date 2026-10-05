"""FastAPI dependency injection utilities."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from app.core.config import settings
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
    """Return the LLM gateway built from settings (overridable in tests)."""
    return AIGateway(settings)


Gateway = Annotated[AIGateway, Depends(get_gateway)]
