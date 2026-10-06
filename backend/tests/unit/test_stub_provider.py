"""Tests that StubProvider works without API keys."""

from __future__ import annotations

import pytest

from ai.providers.stub_provider import StubProvider


@pytest.fixture
def stub() -> StubProvider:
    return StubProvider()


@pytest.mark.asyncio
async def test_stub_health_check(stub):
    assert await stub.health_check() is True


@pytest.mark.asyncio
async def test_stub_complete_returns_string(stub):
    result = await stub.complete("Tell me something")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_stub_complete_format_task(stub):
    result = await stub.complete("format this", system="format")
    assert "STUB" in result


@pytest.mark.asyncio
async def test_stub_complete_decide_task(stub):
    result = await stub.complete("Should I keep this model?", system="decide the experiment")
    assert "STUB" in result


@pytest.mark.asyncio
async def test_stub_complete_structured(stub):
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "confidence": {"type": "number"},
            "reasons": {"type": "array"},
        },
    }
    result = await stub.complete_structured("make a decision", schema)
    assert isinstance(result, dict)
    assert "decision" in result
    assert "confidence" in result
    assert "reasons" in result


@pytest.mark.asyncio
async def test_stub_complete_structured_number_field(stub):
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}
    result = await stub.complete_structured("score this", schema)
    assert isinstance(result["score"], (int, float))


def test_stub_estimate_cost_is_zero(stub):
    cost = stub.estimate_cost(1000, 500)
    assert cost == 0.0


@pytest.mark.asyncio
async def test_stub_list_models(stub):
    models = await stub.list_models()
    assert len(models) == 1
    assert models[0].name == "stub-default"
    assert models[0].is_free is True
    assert models[0].provider == "stub"
    assert models[0].cost_per_1k_prompt_tokens == 0.0


@pytest.mark.asyncio
async def test_stub_no_api_key_needed():
    # Should not raise anything — no api_key argument required
    provider = StubProvider()
    result = await provider.complete("test")
    assert isinstance(result, str)
