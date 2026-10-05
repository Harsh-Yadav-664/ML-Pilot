"""An AIGateway with only the offline stub provider, for tests."""
from __future__ import annotations

from ai.gateway import AIGateway


class StubOnlySettings:
    GROQ_API_KEY = None
    GEMINI_API_KEY = None
    NVIDIA_API_KEY = None
    OPENROUTER_API_KEY = None
    CEREBRAS_API_KEY = None
    MISTRAL_API_KEY = None
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None


def stub_gateway() -> AIGateway:
    gateway = AIGateway(config=StubOnlySettings())
    assert gateway.list_providers() == ["stub"]
    return gateway
