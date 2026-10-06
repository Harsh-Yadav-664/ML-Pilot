"""OllamaProvider: a local model served by Ollama (https://ollama.com).

Nothing leaves the machine, which suits companies that can't send schema
information to a hosted LLM. Registered when OLLAMA_BASE_URL is set.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ml.core.interfaces import AIProvider, ModelInfo


class OllamaProvider(AIProvider):
    name = "ollama"
    DEFAULT_MODEL = "llama3.1"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport  # tests pass httpx.MockTransport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport
        )

    async def _chat(
        self,
        prompt: str,
        system: str,
        model: str | None,
        max_tokens: int,
        temperature: float,
        fmt: Any = None,
    ) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if fmt is not None:
            body["format"] = fmt
        async with self._client() as client:
            resp = await client.post("/api/chat", json=body)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        return await self._chat(prompt, system, model, max_tokens, temperature)

    async def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        text = await self._chat(prompt, system, model, max_tokens, 0.0, fmt=schema)
        return json.loads(text)  # a parse failure raises; the gateway records the fallback

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str | None = None
    ) -> float:
        return 0.0  # runs locally

    async def list_models(self) -> list[ModelInfo]:
        async with self._client() as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
        return [
            ModelInfo(
                name=m["name"],
                provider="ollama",
                context_window=8192,
                cost_per_1k_prompt_tokens=0.0,
                cost_per_1k_completion_tokens=0.0,
                capabilities=["complete", "structured"],
                is_free=True,
            )
            for m in resp.json().get("models", [])
        ]

    async def health_check(self) -> bool:
        try:
            await self.list_models()
            return True
        except httpx.HTTPError:
            return False
