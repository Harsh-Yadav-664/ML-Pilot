"""An AIGateway with only the offline stub provider, for tests."""

from __future__ import annotations

from ai.context_builder import BuiltPrompt, ContextBuilder
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


def prompt(text: str, system: str = "", purpose: str = "test") -> BuiltPrompt:
    """A plain-text prompt for tests that exercise the gateway itself."""
    return ContextBuilder().prompt(purpose, system).text("", text).build()


def recording_gateway() -> AIGateway:
    """The stub-only gateway with the prompt log, writing to the test database and projects dir."""
    from app.services.llm_gateway import make_gateway

    gateway = make_gateway()
    stub = AIGateway(config=StubOnlySettings(), recorder=gateway.recorder)
    assert stub.list_providers() == ["stub"]
    return stub
