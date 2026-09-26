"""Tests for AIGateway registration and fallback."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from ai.gateway import AIGateway
from ai.router import TaskType


class MockSettings:
    GROQ_API_KEY = None
    GEMINI_API_KEY = None
    NVIDIA_API_KEY = None
    OPENROUTER_API_KEY = None
    CEREBRAS_API_KEY = None
    MISTRAL_API_KEY = None
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None


@pytest.fixture
def gateway():
    return AIGateway(config=MockSettings())


def test_gateway_always_has_stub(gateway):
    assert "stub" in gateway.list_providers()


def test_gateway_list_providers_returns_list(gateway):
    providers = gateway.list_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1


def test_gateway_no_real_providers_when_no_keys(gateway):
    providers = gateway.list_providers()
    # Without any API keys, only stub should be registered
    assert providers == ["stub"]


@pytest.mark.asyncio
async def test_gateway_complete_uses_stub_fallback(gateway):
    result = await gateway.complete(
        task_type=TaskType.FORMAT,
        prompt="Format this output",
        system="format",
    )
    assert isinstance(result, str)
    assert "STUB" in result


@pytest.mark.asyncio
async def test_gateway_complete_summarize(gateway):
    result = await gateway.complete(
        task_type=TaskType.SUMMARIZE,
        prompt="Summarize the experiment results.",
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_gateway_complete_decide(gateway):
    result = await gateway.complete(
        task_type=TaskType.DECIDE,
        prompt="Should we keep this experiment? F1 improved from 0.80 to 0.85.",
        system="decide",
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_gateway_fallback_on_provider_error(gateway):
    """When first provider fails, gateway should fall back to next (stub)."""
    from ai.providers.stub_provider import StubProvider

    # Add a failing mock provider at the front
    failing_provider = MagicMock()
    failing_provider.complete = AsyncMock(side_effect=Exception("API down"))
    failing_provider.estimate_cost = MagicMock(return_value=0.0)
    gateway.providers = {"failing": failing_provider, "stub": StubProvider()}

    # Patch router to return failing first, then stub
    gateway.router.get_ordered_providers = MagicMock(
        return_value=[("failing", "some-model"), ("stub", "stub-default")]
    )

    result = await gateway.complete(TaskType.FORMAT, "test")
    assert isinstance(result, str)


def test_gateway_cost_summary(gateway):
    summary = gateway.get_cost_summary()
    assert "total_cost_usd" in summary
    assert "total_requests" in summary
    assert summary["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_gateway_complete_tracks_cost(gateway):
    await gateway.complete(TaskType.FORMAT, "test prompt")
    summary = gateway.get_cost_summary()
    assert summary["total_requests"] == 1


@pytest.mark.asyncio
async def test_gateway_all_task_types(gateway):
    for task_type in TaskType:
        result = await gateway.complete(task_type, f"Task: {task_type.value}")
        assert isinstance(result, str)
