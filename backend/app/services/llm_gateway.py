"""The LLM gateway the app uses: configured providers plus the prompt log."""

from __future__ import annotations

from ai.gateway import AIGateway
from app.core import datasets
from app.core.config import settings
from app.db import session as db_session
from app.services.prompt_log import PromptLog


def make_gateway() -> AIGateway:
    return AIGateway(
        settings, recorder=PromptLog(lambda: db_session.AsyncSessionLocal(), datasets.PROJECTS_DIR)
    )
