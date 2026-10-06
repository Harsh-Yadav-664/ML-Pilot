"""AIGateway — registers providers, routes each task by config, records every call.

Every call returns (or records) an LLMResult: which provider and model answered,
estimated tokens, cost, latency, and decision_mode: 'llm' when a real provider
answered, 'fallback' when the offline stub did (nothing configured, or every
configured provider failed). Every prompt passes through the prompt hooks.
"""
from __future__ import annotations

import importlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ai.cost_tracker import CostTracker
from ai.providers.stub_provider import StubProvider
from ai.router import TaskRouter, TaskType
from ml.core.interfaces import AIProvider, ModelInfo

logger = logging.getLogger(__name__)

# provider name -> (settings attribute that enables it, module, class)
PROVIDER_FACTORIES: dict[str, tuple[str, str, str]] = {
    "groq": ("GROQ_API_KEY", "ai.providers.groq_provider", "GroqProvider"),
    "gemini": ("GEMINI_API_KEY", "ai.providers.gemini_provider", "GeminiProvider"),
    "nvidia_nim": ("NVIDIA_API_KEY", "ai.providers.nvidia_nim_provider", "NvidiaNIMProvider"),
    "openrouter": ("OPENROUTER_API_KEY", "ai.providers.openrouter_provider", "OpenRouterProvider"),
    "cerebras": ("CEREBRAS_API_KEY", "ai.providers.cerebras_provider", "CerebrasProvider"),
    "mistral": ("MISTRAL_API_KEY", "ai.providers.mistral_provider", "MistralProvider"),
    "openai": ("OPENAI_API_KEY", "ai.providers.openai_provider", "OpenAIProvider"),
    "anthropic": ("ANTHROPIC_API_KEY", "ai.providers.anthropic_provider", "AnthropicProvider"),
    "ollama": ("OLLAMA_BASE_URL", "ai.providers.ollama_provider", "OllamaProvider"),
}

OFFLINE_NOTE = "offline stub provider"


@dataclass
class LLMResult:
    """One answered LLM call. Token counts are word-count estimates."""

    task_type: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    decision_mode: str  # 'llm' | 'fallback'
    prompt_id: str
    text: str = ""
    data: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)  # providers tried before this one

    def meta(self) -> dict[str, Any]:
        """Everything but the content, for recording on runs and API responses."""
        out = asdict(self)
        out.pop("text")
        out.pop("data")
        if self.provider == "stub":
            out["note"] = OFFLINE_NOTE
        return out


@dataclass
class PromptRecord:
    prompt_id: str
    task_type: str
    system: str
    prompt: str
    provider: str
    model: str


def _log_prompt(record: PromptRecord) -> None:
    # Content is not logged here (it can hold schema details); #48 stores prompts properly.
    logger.info("LLM prompt %s task=%s provider=%s model=%s chars=%d",
                record.prompt_id, record.task_type, record.provider, record.model,
                len(record.system) + len(record.prompt))


class AIGateway:
    """Central AI gateway: registers providers, routes requests, records fallbacks.

    Providers register only when their key (or Ollama URL) is configured.
    StubProvider is always registered as the last-resort fallback.
    """

    def __init__(self, config: Any, router: TaskRouter | None = None) -> None:
        self.providers: dict[str, AIProvider] = {}
        self.router = router or TaskRouter()
        self.cost_tracker = CostTracker()
        self.calls: list[LLMResult] = []
        self.prompt_hooks: list[Callable[[PromptRecord], None]] = [_log_prompt]
        self._register_providers(config)

    def _register_providers(self, config: Any) -> None:
        """Register every configured provider; one log line for each one skipped."""
        for name, (setting, module, cls_name) in PROVIDER_FACTORIES.items():
            value = getattr(config, setting, None)
            if not value:
                logger.info("LLM provider %s skipped: %s not set", name, setting)
                continue
            try:
                cls = getattr(importlib.import_module(module), cls_name)
                self.providers[name] = cls(value)
                logger.info("Registered LLM provider: %s", name)
            except Exception as e:
                logger.warning("LLM provider %s could not be registered: %s", name, e)
        self.providers["stub"] = StubProvider()
        if len(self.providers) == 1:
            logger.info("No LLM provider configured: using the %s", OFFLINE_NOTE)

    def add_prompt_hook(self, hook: Callable[[PromptRecord], None]) -> None:
        """Call *hook* with every prompt sent to any provider (used by prompt logging, #48)."""
        self.prompt_hooks.append(hook)

    async def _dispatch(self, task_type: TaskType, prompt: str, system: str, model: str | None,
                        call: Callable[[AIProvider, str], Any], structured: bool,
                        experiment_id: str | None = None, project_id: str | None = None) -> LLMResult:
        ordered = self.router.get_ordered_providers(task_type, list(self.providers.keys()))
        prompt_id = uuid.uuid4().hex[:12]
        errors: list[str] = []
        for provider_name, default_model in ordered:
            provider = self.providers[provider_name]
            selected_model = model or default_model
            for hook in self.prompt_hooks:
                hook(PromptRecord(prompt_id, task_type.value, system, prompt, provider_name, selected_model))
            start = time.perf_counter()
            try:
                answer = await call(provider, selected_model)
            except Exception as e:
                errors.append(f"{provider_name}: {e}")
                logger.warning("Provider %r failed for task %r: %s. Trying next...", provider_name, task_type.value, e)
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            text = answer if isinstance(answer, str) else str(answer)
            tokens_in, tokens_out = len((system + " " + prompt).split()), len(text.split())
            cost = provider.estimate_cost(prompt_tokens=tokens_in, completion_tokens=tokens_out, model=selected_model)
            self.cost_tracker.log(provider=provider_name, model=selected_model, prompt_tokens=tokens_in,
                                  completion_tokens=tokens_out, cost_usd=cost, task_type=task_type.value,
                                  experiment_id=experiment_id, project_id=project_id)
            result = LLMResult(
                task_type=task_type.value, provider=provider_name, model=selected_model,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, latency_ms=round(latency_ms, 1),
                decision_mode="fallback" if provider_name == "stub" else "llm", prompt_id=prompt_id,
                text="" if structured else text, data=answer if structured else None, errors=errors,
            )
            self.calls.append(result)
            return result
        raise RuntimeError(f"All providers failed for task {task_type.value!r}: {'; '.join(errors)}")

    async def complete_result(self, task_type: TaskType, prompt: str, system: str = "",
                              model: str | None = None, max_tokens: int = 1024, temperature: float = 0.7,
                              experiment_id: str | None = None, project_id: str | None = None) -> LLMResult:
        """Text completion with fallback through the routing order; returns the full record."""
        async def call(provider: AIProvider, selected_model: str):
            return await provider.complete(prompt=prompt, system=system, model=selected_model,
                                           max_tokens=max_tokens, temperature=temperature)
        return await self._dispatch(task_type, prompt, system, model, call, False, experiment_id, project_id)

    async def complete_structured_result(self, task_type: TaskType, prompt: str, schema: dict[str, Any],
                                         system: str = "", model: str | None = None,
                                         max_tokens: int = 2048) -> LLMResult:
        """Structured (JSON) completion; a provider whose output doesn't parse counts as failed."""
        async def call(provider: AIProvider, selected_model: str):
            data = await provider.complete_structured(prompt=prompt, schema=schema, system=system,
                                                      model=selected_model, max_tokens=max_tokens)
            if not isinstance(data, dict):
                raise ValueError(f"structured output is {type(data).__name__}, not an object")
            return data
        return await self._dispatch(task_type, prompt, system, model, call, True)

    async def complete(self, task_type: TaskType, prompt: str, system: str = "", model: str | None = None,
                       max_tokens: int = 1024, temperature: float = 0.7, experiment_id: str | None = None,
                       project_id: str | None = None) -> str:
        """Text only; see complete_result for provider, cost and decision_mode."""
        return (await self.complete_result(task_type, prompt, system, model, max_tokens, temperature,
                                           experiment_id, project_id)).text

    async def complete_structured(self, task_type: TaskType, prompt: str, schema: dict[str, Any],
                                  system: str = "", model: str | None = None,
                                  max_tokens: int = 2048) -> dict[str, Any]:
        """Data only; see complete_structured_result for provider, cost and decision_mode."""
        return (await self.complete_structured_result(task_type, prompt, schema, system, model, max_tokens)).data

    def list_providers(self) -> list[str]:
        """Return the names of all registered providers."""
        return list(self.providers.keys())

    async def list_all_models(self) -> dict[str, list[ModelInfo]]:
        """Return models from all registered providers."""
        result: dict[str, list[ModelInfo]] = {}
        for name, provider in self.providers.items():
            try:
                result[name] = await provider.list_models()
            except Exception:
                result[name] = []
        return result

    def get_cost_summary(self) -> dict:
        """Return a cost usage summary."""
        return self.cost_tracker.summary()
