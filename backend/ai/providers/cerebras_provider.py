"""CerebrasProvider — ultra-fast inference via Cerebras."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)


_CEREBRAS_MODELS = [
    ModelInfo(
        name="llama3.1-8b",
        provider="cerebras",
        context_window=8192,
        cost_per_1k_prompt_tokens=0.10,
        cost_per_1k_completion_tokens=0.10,
        capabilities=["complete", "format", "summarize"],
        is_free=False,
    ),
    ModelInfo(
        name="llama3.1-70b",
        provider="cerebras",
        context_window=8192,
        cost_per_1k_prompt_tokens=0.60,
        cost_per_1k_completion_tokens=0.60,
        capabilities=["complete", "structured", "summarize", "analyze", "report"],
        is_free=False,
    ),
]


class CerebrasProvider(AIProvider):
    """Cerebras provider using the OpenAI-compatible API."""

    BASE_URL = "https://api.cerebras.ai/v1"
    DEFAULT_MODEL = "llama3.1-8b"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = []
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
        sys_json = f"{system}\n\nRespond ONLY with valid JSON: {json.dumps(schema)}"
        raw = await self.complete(
            prompt, system=sys_json, model=model, max_tokens=max_tokens, temperature=0.1
        )
        if "```" in raw:
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json")
        return json.loads(raw.strip())

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str | None = None
    ) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _CEREBRAS_MODELS if mi.name == m), _CEREBRAS_MODELS[0])
        return (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens + (
            completion_tokens / 1000
        ) * info.cost_per_1k_completion_tokens

    async def list_models(self) -> list[ModelInfo]:
        return list(_CEREBRAS_MODELS)

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:  # health check: any client error means unhealthy
            logger.warning("%s health check failed", type(self).__name__, exc_info=True)
            return False
