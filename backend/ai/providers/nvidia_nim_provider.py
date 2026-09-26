"""NvidiaNIMProvider — NVIDIA NIM via OpenAI-compatible API."""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import AsyncOpenAI

from ml.core.interfaces import AIProvider, ModelInfo

_NVIDIA_MODELS = [
    ModelInfo(
        name="meta/llama-3.1-8b-instruct",
        provider="nvidia_nim",
        context_window=131072,
        cost_per_1k_prompt_tokens=0.20,
        cost_per_1k_completion_tokens=0.20,
        capabilities=["complete", "structured", "format", "summarize", "analyze"],
        is_free=False,
    ),
    ModelInfo(
        name="meta/llama-3.1-70b-instruct",
        provider="nvidia_nim",
        context_window=131072,
        cost_per_1k_prompt_tokens=0.97,
        cost_per_1k_completion_tokens=0.97,
        capabilities=["complete", "structured", "summarize", "hypothesize", "analyze", "report", "decide"],
        is_free=False,
    ),
]


class NvidiaNIMProvider(AIProvider):
    """NVIDIA NIM provider using the OpenAI-compatible API."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL)

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
        system_with_json = f"{system}\n\nRespond ONLY with valid JSON: {json.dumps(schema)}"
        raw = await self.complete(prompt, system=system_with_json, model=model, max_tokens=max_tokens, temperature=0.1)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _NVIDIA_MODELS if mi.name == m), _NVIDIA_MODELS[0])
        return (
            (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens
            + (completion_tokens / 1000) * info.cost_per_1k_completion_tokens
        )

    async def list_models(self) -> list[ModelInfo]:
        return list(_NVIDIA_MODELS)

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
