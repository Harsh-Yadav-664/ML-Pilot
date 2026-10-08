"""The scripted provider used by the end-to-end test: it replays answers and is labelled as such."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.providers.scripted_provider import ENV_VAR, ScriptedProvider
from ai.router import TaskType
from app.services.llm_gateway import make_gateway

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "stub_feature_proposals.yaml"


async def test_it_replays_the_named_answers_in_order_then_runs_out() -> None:
    provider = ScriptedProvider(f"{FIXTURE}:good_refunds_14d,leaky_sql")
    first = await provider.complete_structured("p", {})
    assert first["name"] == "refund_count_14d"
    assert (await provider.complete_structured("p", {}))["name"]
    with pytest.raises(RuntimeError, match="no more answers"):
        await provider.complete_structured("p", {})


def test_a_malformed_script_is_refused() -> None:
    with pytest.raises(ValueError, match="must look like"):
        ScriptedProvider("no-keys")


async def test_the_gateway_uses_it_for_sql_only_and_labels_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_VAR, f"{FIXTURE}:good_refunds_14d")
    gateway = make_gateway()
    order = gateway.router.get_ordered_providers(TaskType.SQL, list(gateway.providers))
    assert order[0][0] == "scripted" and order[-1][0] == "stub"
    other = gateway.router.get_ordered_providers(TaskType.SPEC, list(gateway.providers))
    assert all(p != "scripted" for p, _ in other)
    monkeypatch.delenv(ENV_VAR)
    assert "scripted" not in make_gateway().providers
