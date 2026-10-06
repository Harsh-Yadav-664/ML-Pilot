"""AnthropicProvider — Anthropic Claude (paid, future integration)."""

from __future__ import annotations

import json
import logging
from typing import Any

from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)


_ANTHROPIC_MODELS = [
    ModelInfo(
        name="claude-3-haiku-20240307",
        provider="anthropic",
        context_window=200000,
        cost_per_1k_prompt_tokens=0.25,
        cost_per_1k_completion_tokens=1.25,
        capabilities=["complete", "structured", "format", "summarize", "analyze"],
        is_free=False,
    ),
    ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        context_window=200000,
        cost_per_1k_prompt_tokens=3.00,
        cost_per_1k_completion_tokens=15.00,
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
]


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider.

    NOTE: Uses the anthropic SDK directly (not OpenAI-compatible).
    The anthropic package is optional — import is deferred.
    """

    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from e
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        return response.content[0].text

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
        info = next((mi for mi in _ANTHROPIC_MODELS if mi.name == m), _ANTHROPIC_MODELS[0])
        return (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens + (
            completion_tokens / 1000
        ) * info.cost_per_1k_completion_tokens

    async def list_models(self) -> list[ModelInfo]:
        return list(_ANTHROPIC_MODELS)

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:  # health check: any client error means unhealthy
            logger.warning("%s health check failed", type(self).__name__, exc_info=True)
            return False
