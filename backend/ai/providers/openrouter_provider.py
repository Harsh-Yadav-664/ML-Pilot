"""OpenRouterProvider — aggregated model routing via OpenRouter."""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import AsyncOpenAI

from ml.core.interfaces import AIProvider, ModelInfo

_OPENROUTER_MODELS = [
    ModelInfo(
        name="meta-llama/llama-3.1-8b-instruct:free",
        provider="openrouter",
        context_window=131072,
        cost_per_1k_prompt_tokens=0.0,
        cost_per_1k_completion_tokens=0.0,
        capabilities=["complete", "summarize", "format"],
        is_free=True,
    ),
    ModelInfo(
        name="google/gemma-2-9b-it:free",
        provider="openrouter",
        context_window=8192,
        cost_per_1k_prompt_tokens=0.0,
        cost_per_1k_completion_tokens=0.0,
        capabilities=["complete", "summarize"],
        is_free=True,
    ),
    ModelInfo(
        name="mistralai/mistral-7b-instruct:free",
        provider="openrouter",
        context_window=32768,
        cost_per_1k_prompt_tokens=0.0,
        cost_per_1k_completion_tokens=0.0,
        capabilities=["complete", "summarize", "format"],
        is_free=True,
    ),
]


class OpenRouterProvider(AIProvider):
    """OpenRouter aggregated model provider. Supports free and paid models."""

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

    def __init__(self, api_key: str, site_url: str = "https://mlpilot.dev", site_name: str = "MLPilot") -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
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
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        sys_json = f"{system}\n\nRespond ONLY with valid JSON: {json.dumps(schema)}"
        raw = await self.complete(prompt, system=sys_json, model=model, max_tokens=max_tokens, temperature=0.1)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _OPENROUTER_MODELS if mi.name == m), _OPENROUTER_MODELS[0])
        return (
            (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens
            + (completion_tokens / 1000) * info.cost_per_1k_completion_tokens
        )

    async def list_models(self) -> list[ModelInfo]:
        return list(_OPENROUTER_MODELS)

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
