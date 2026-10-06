"""OpenAIProvider — OpenAI API (paid, future integration)."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)


_OPENAI_MODELS = [
    ModelInfo(
        name="gpt-4o-mini",
        provider="openai",
        context_window=128000,
        cost_per_1k_prompt_tokens=0.15,
        cost_per_1k_completion_tokens=0.60,
        capabilities=["complete", "structured", "format", "summarize", "analyze", "report"],
        is_free=False,
    ),
    ModelInfo(
        name="gpt-4o",
        provider="openai",
        context_window=128000,
        cost_per_1k_prompt_tokens=2.50,
        cost_per_1k_completion_tokens=10.00,
        capabilities=["complete", "structured", "format", "summarize", "hypothesize", "analyze", "synthesize", "report", "decide"],
        is_free=False,
    ),
]


class OpenAIProvider(AIProvider):
    """OpenAI API provider."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", model: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content or ""

    async def complete_structured(self, prompt: str, schema: dict[str, Any], system: str = "", model: Optional[str] = None, max_tokens: int = 2048) -> dict[str, Any]:
        sys_json = f"{system}\n\nRespond ONLY with valid JSON: {json.dumps(schema)}"
        raw = await self.complete(prompt, system=sys_json, model=model, max_tokens=max_tokens, temperature=0.1)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _OPENAI_MODELS if mi.name == m), _OPENAI_MODELS[0])
        return (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens + (completion_tokens / 1000) * info.cost_per_1k_completion_tokens

    async def list_models(self) -> list[ModelInfo]:
        return list(_OPENAI_MODELS)

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:  # noqa: BLE001 - health check: any client error means unhealthy
            logger.warning("%s health check failed", type(self).__name__, exc_info=True)
            return False
