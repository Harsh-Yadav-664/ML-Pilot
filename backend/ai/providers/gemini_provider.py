"""GeminiProvider — Google Gemini AI provider."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)


_GEMINI_MODELS = [
    ModelInfo(
        name="gemini-1.5-flash",
        provider="gemini",
        context_window=1000000,
        cost_per_1k_prompt_tokens=0.075,
        cost_per_1k_completion_tokens=0.30,
        capabilities=["complete", "structured", "format", "summarize", "analyze", "report"],
        is_free=False,
    ),
    ModelInfo(
        name="gemini-1.5-pro",
        provider="gemini",
        context_window=2000000,
        cost_per_1k_prompt_tokens=1.25,
        cost_per_1k_completion_tokens=5.00,
        capabilities=["complete", "structured", "format", "summarize", "hypothesize", "analyze", "synthesize", "report", "decide"],
        is_free=False,
    ),
]


class GeminiProvider(AIProvider):
    """Google Gemini AI provider using the google-generativeai SDK."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str) -> None:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._api_key = api_key

    def _get_model(self, model_name: Optional[str] = None):
        return self._genai.GenerativeModel(model_name or self.DEFAULT_MODEL)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        import asyncio
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        m = self._get_model(model)
        response = await asyncio.to_thread(
            m.generate_content,
            full_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return response.text

    async def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        system_with_json = f"{system}\n\nRespond ONLY with valid JSON matching this schema: {json.dumps(schema)}"
        raw = await self.complete(prompt, system=system_with_json, model=model, max_tokens=max_tokens, temperature=0.2)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        m = model or self.DEFAULT_MODEL
        info = next((mi for mi in _GEMINI_MODELS if mi.name == m), _GEMINI_MODELS[0])
        return (
            (prompt_tokens / 1000) * info.cost_per_1k_prompt_tokens
            + (completion_tokens / 1000) * info.cost_per_1k_completion_tokens
        )

    async def list_models(self) -> list[ModelInfo]:
        return list(_GEMINI_MODELS)

    async def health_check(self) -> bool:
        try:
            m = self._get_model()
            import asyncio
            response = await asyncio.to_thread(m.generate_content, "ping")
            return bool(response.text)
        except Exception:  # noqa: BLE001 - health check: any client error means unhealthy
            logger.warning("%s health check failed", type(self).__name__, exc_info=True)
            return False
