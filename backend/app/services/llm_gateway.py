"""The LLM gateway the app uses: configured providers plus the prompt log."""

from __future__ import annotations

import os

from ai.gateway import AIGateway
from ai.providers.scripted_provider import ENV_VAR, MODEL, ScriptedProvider
from app.core import datasets
from app.core.config import settings
from app.db import session as db_session
from app.services.prompt_log import PromptLog


def make_gateway() -> AIGateway:
    gateway = AIGateway(
        settings, recorder=PromptLog(lambda: db_session.AsyncSessionLocal(), datasets.PROJECTS_DIR)
    )
    script = os.environ.get(ENV_VAR)
    if script:  # the end-to-end test replays fixed feature proposals; see the provider's docstring
        gateway.providers["scripted"] = ScriptedProvider(script)
        gateway.router.routing["sql"].insert(0, ("scripted", MODEL))
    return gateway
