"""GroqProvider — OpenAI-compatible provider via Groq."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)


_GROQ_MODELS = [
    ModelInfo(
        name="llama-3.1-8b-instant",
        provider="groq",
        context_window=131072,
        cost_per_1k_prompt_tokens=0.05,
        cost_per_1k_completion_tokens=0.08,
        capabilities=["complete", "structured", "format", "summarize", "hypothesize", "analyze"],
        is_free=False,
    ),
    ModelInfo(
        name="llama-3.3-70b-versatile",
        provider="groq",
        context_window=131072,
        cost_per_1k_prompt_tokens=0.59,
        cost_per_1k_completion_tokens=0.79,
        capabilities=[
            "complete",
            "structured",
            "format",
            "summarize",
            "hypothesize",
            "analyze",
            "synthesize",
            "report",
            "decide",
        ],
        is_free=False,
    ),
    ModelInfo(
        name="mixtral-8x7b-32768",
        provider="groq",
        context_window=32768,
        cost_per_1k_prompt_tokens=0.27,
        cost_per_1k_completion_tokens=0.27,
        capabilities=["complete", "structured", "summarize", "analyze", "report"],
        is_free=False,
    ),
]


class GroqProvider(AIProvider):
    """Groq LLM provider using the OpenAI-compatible API."""

    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL)
        self._api_key = api_key

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        system_with_json = (
            f"{system}\n\nRespond ONLY with valid JSON matching this schema: {json.dumps(schema)}"
        )
        raw = await self.complete(
            prompt, system=system_with_json, model=model, max_tokens=max_tokens, temperature=0.2
        )
        # Extract JSON from markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json")
        return json.loads(raw.strip())

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str | None = None
    ) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _GROQ_MODELS if mi.name == m), _GROQ_MODELS[0])
        return (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens + (
            completion_tokens / 1000
        ) * info.cost_per_1k_completion_tokens

    async def list_models(self) -> list[ModelInfo]:
        return list(_GROQ_MODELS)

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return True
        except Exception:  # health check: any client error means unhealthy
            logger.warning("%s health check failed", type(self).__name__, exc_info=True)
            return False
