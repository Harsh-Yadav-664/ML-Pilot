"""AIGateway — registers all providers and dispatches requests with fallback."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ml.core.interfaces import AIProvider, ModelInfo
from ai.cost_tracker import CostTracker
from ai.router import TaskRouter, TaskType
from ai.providers.stub_provider import StubProvider

logger = logging.getLogger(__name__)


class AIGateway:
    """Central AI gateway: registers providers, routes requests, handles fallbacks.

    Provider registration is conditional on API key availability.
    StubProvider is ALWAYS registered as the last-resort fallback.
    """

    def __init__(self, config: Any) -> None:
        """Initialize gateway with application Settings.

        Args:
            config: pydantic-settings Settings instance.
        """
        self.providers: dict[str, AIProvider] = {}
        self.router = TaskRouter()
        self.cost_tracker = CostTracker()
        self._register_providers(config)

    def _register_providers(self, config: Any) -> None:
        """Register all configured providers. Always registers StubProvider last."""
        # Groq
        if getattr(config, "GROQ_API_KEY", None):
            try:
                from ai.providers.groq_provider import GroqProvider
                self.providers["groq"] = GroqProvider(config.GROQ_API_KEY)
                logger.info("Registered: GroqProvider")
            except Exception as e:
                logger.warning(f"Failed to register GroqProvider: {e}")

        # Gemini
        if getattr(config, "GEMINI_API_KEY", None):
            try:
                from ai.providers.gemini_provider import GeminiProvider
                self.providers["gemini"] = GeminiProvider(config.GEMINI_API_KEY)
                logger.info("Registered: GeminiProvider")
            except Exception as e:
                logger.warning(f"Failed to register GeminiProvider: {e}")

        # NVIDIA NIM
        if getattr(config, "NVIDIA_API_KEY", None):
            try:
                from ai.providers.nvidia_nim_provider import NvidiaNIMProvider
                self.providers["nvidia_nim"] = NvidiaNIMProvider(config.NVIDIA_API_KEY)
                logger.info("Registered: NvidiaNIMProvider")
            except Exception as e:
                logger.warning(f"Failed to register NvidiaNIMProvider: {e}")

        # OpenRouter
        if getattr(config, "OPENROUTER_API_KEY", None):
            try:
                from ai.providers.openrouter_provider import OpenRouterProvider
                self.providers["openrouter"] = OpenRouterProvider(config.OPENROUTER_API_KEY)
                logger.info("Registered: OpenRouterProvider")
            except Exception as e:
                logger.warning(f"Failed to register OpenRouterProvider: {e}")

        # Cerebras
        if getattr(config, "CEREBRAS_API_KEY", None):
            try:
                from ai.providers.cerebras_provider import CerebrasProvider
                self.providers["cerebras"] = CerebrasProvider(config.CEREBRAS_API_KEY)
                logger.info("Registered: CerebrasProvider")
            except Exception as e:
                logger.warning(f"Failed to register CerebrasProvider: {e}")

        # Mistral
        if getattr(config, "MISTRAL_API_KEY", None):
            try:
                from ai.providers.mistral_provider import MistralProvider
                self.providers["mistral"] = MistralProvider(config.MISTRAL_API_KEY)
                logger.info("Registered: MistralProvider")
            except Exception as e:
                logger.warning(f"Failed to register MistralProvider: {e}")

        # OpenAI
        if getattr(config, "OPENAI_API_KEY", None):
            try:
                from ai.providers.openai_provider import OpenAIProvider
                self.providers["openai"] = OpenAIProvider(config.OPENAI_API_KEY)
                logger.info("Registered: OpenAIProvider")
            except Exception as e:
                logger.warning(f"Failed to register OpenAIProvider: {e}")

        # StubProvider — always registered last
        self.providers["stub"] = StubProvider()
        logger.info("Registered: StubProvider (offline fallback)")

    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        experiment_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """Dispatch a completion request with automatic fallback.

        Tries providers in priority order for the task type.
        Falls back to the next provider on any error.
        """
        ordered = self.router.get_ordered_providers(
            task_type, list(self.providers.keys())
        )

        last_error: Optional[Exception] = None
        for provider_name, default_model in ordered:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            selected_model = model or default_model
            try:
                result = await provider.complete(
                    prompt=prompt,
                    system=system,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Estimate cost and log
                cost = provider.estimate_cost(
                    prompt_tokens=len(prompt.split()),  # rough estimate
                    completion_tokens=len(result.split()),
                    model=selected_model,
                )
                self.cost_tracker.log(
                    provider=provider_name,
                    model=selected_model,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(result.split()),
                    cost_usd=cost,
                    task_type=task_type.value,
                    experiment_id=experiment_id,
                    project_id=project_id,
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name!r} failed for task {task_type.value!r}: {e}. Trying next...")
                continue

        raise RuntimeError(
            f"All providers failed for task {task_type.value!r}. Last error: {last_error}"
        )

    async def complete_structured(
        self,
        task_type: TaskType,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Dispatch a structured completion request with fallback."""
        ordered = self.router.get_ordered_providers(
            task_type, list(self.providers.keys())
        )
        last_error: Optional[Exception] = None
        for provider_name, default_model in ordered:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            selected_model = model or default_model
            try:
                return await provider.complete_structured(
                    prompt=prompt,
                    schema=schema,
                    system=system,
                    model=selected_model,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name!r} structured call failed: {e}. Trying next...")
                continue
        raise RuntimeError(f"All providers failed for structured task {task_type.value!r}. Last error: {last_error}")

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
