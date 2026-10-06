"""Tests for DataCleaningAgent with the offline stub provider."""

from __future__ import annotations

import pytest

from ai.gateway import AIGateway
from ml.agents.cleaning_agent import CleaningStrategyError, DataCleaningAgent
from ml.core.interfaces import ProfileResult
from tests.fixtures.gateway import StubOnlySettings


@pytest.fixture
def profile() -> ProfileResult:
    return ProfileResult(
        rows=4,
        columns=3,
        missing_rate=0.0,
        duplicate_rows=0,
        target_balance={},
        column_stats={
            "age": {"dtype": "int64", "missing_count": 0, "missing_rate": 0.0, "unique_count": 4},
            "city": {"dtype": "object", "missing_count": 0, "missing_rate": 0.0, "unique_count": 2},
            "target": {
                "dtype": "int64",
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": 2,
            },
        },
    )


def agent_returning(result):
    gateway = AIGateway(config=StubOnlySettings())

    async def fake(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    gateway.providers["stub"].complete_structured = fake
    return DataCleaningAgent(gateway)


async def test_stub_provider_returns_valid_strategy(profile):
    agent = DataCleaningAgent(AIGateway(config=StubOnlySettings()))
    strategy = await agent.generate_cleaning_strategy(profile, "target")
    assert strategy == {"columns": {}}


async def test_valid_llm_strategy_is_returned(profile):
    columns = {
        "age": ["impute_median", "standard_scale"],
        "city": ["impute_constant", "onehot_encode"],
    }
    strategy = await agent_returning({"columns": columns}).generate_cleaning_strategy(
        profile, "target"
    )
    assert strategy == {"columns": columns}


async def test_provider_failure_raises(profile):
    with pytest.raises(CleaningStrategyError, match="LLM call failed"):
        await agent_returning(RuntimeError("down")).generate_cleaning_strategy(profile, "target")


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"columns": []},
        {"columns": {"nope": ["impute_median"]}},
        {"columns": {"target": ["impute_median"]}},
        {"columns": {"age": ["drop_table"]}},
        {"columns": {"age": "impute_median"}},
    ],
)
async def test_invalid_llm_strategy_raises(profile, bad):
    with pytest.raises(CleaningStrategyError):
        await agent_returning(bad).generate_cleaning_strategy(profile, "target")
