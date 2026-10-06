"""Config-driven LLM routing: any provider per tier, every call recorded, fallbacks visible."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from ai.gateway import PROVIDER_FACTORIES, AIGateway, PromptRecord
from ai.llm_config import DEFAULT_TIERS, load_routing, validate_tiers
from ai.providers.ollama_provider import OllamaProvider
from ai.router import TaskRouter, TaskType


def _settings(**values):
    keys = {setting: None for setting, _, _ in PROVIDER_FACTORIES.values()}
    return SimpleNamespace(**{**keys, **values})


def _only(provider: str, model: str) -> TaskRouter:
    entry = [f"{provider}:{model}"]
    return TaskRouter(validate_tiers({"cheap": entry, "reasoning": entry, "sql": entry}))


@pytest.mark.parametrize("provider", sorted(PROVIDER_FACTORIES))
async def test_each_registered_provider_can_be_routed_to(provider, monkeypatch):
    setting = PROVIDER_FACTORIES[provider][0]
    value = "http://ollama.test" if provider == "ollama" else "test-key-not-real"
    gateway = AIGateway(_settings(**{setting: value}), router=_only(provider, "model-x"))
    assert set(gateway.providers) == {provider, "stub"}

    seen = {}

    async def fake_complete(prompt, system="", model=None, max_tokens=1024, temperature=0.7):
        seen["model"] = model  # the network call itself is replaced; routing is what is tested
        return "answer from " + provider

    monkeypatch.setattr(gateway.providers[provider], "complete", fake_complete)
    result = await gateway.complete_result(TaskType.SUMMARIZE, "hello")
    assert (result.provider, result.model, result.decision_mode) == (provider, "model-x", "llm")
    assert seen["model"] == "model-x" and result.text == "answer from " + provider
    assert result.tokens_in > 0 and result.tokens_out > 0 and result.latency_ms >= 0


async def test_ollama_provider_over_mocked_http():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        content = json.dumps({"name": "x"}) if "format" in body else "local answer"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    provider = OllamaProvider("http://ollama.test", transport=httpx.MockTransport(handler))
    gateway = AIGateway(_settings(), router=_only("ollama", "llama3.1"))
    gateway.providers = {"ollama": provider, **gateway.providers}

    text = await gateway.complete_result(TaskType.HYPOTHESIZE, "hi", system="be brief")
    data = await gateway.complete_structured_result(TaskType.SQL, "hi", schema={"type": "object"})
    assert text.text == "local answer" and text.provider == "ollama" and text.cost_usd == 0.0
    assert data.data == {"name": "x"} and data.decision_mode == "llm"
    assert requests[0]["model"] == "llama3.1" and requests[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert requests[1]["format"] == {"type": "object"}


async def test_parse_failure_falls_back_to_stub_and_is_recorded(monkeypatch):
    gateway = AIGateway(_settings(OPENAI_API_KEY="test-key-not-real"), router=_only("openai", "gpt-4o"))

    async def bad_json(**kwargs):
        raise json.JSONDecodeError("Expecting value", "not json", 0)

    monkeypatch.setattr(gateway.providers["openai"], "complete_structured", bad_json)
    result = await gateway.complete_structured_result(TaskType.HYPOTHESIZE, "idea?", schema={"properties": {}})
    assert result.provider == "stub" and result.decision_mode == "fallback"
    assert result.errors and result.errors[0].startswith("openai: Expecting value")
    assert result.meta()["note"] == "offline stub provider"
    assert gateway.calls[-1] is result


async def test_every_prompt_goes_through_the_hooks():
    gateway = AIGateway(_settings())
    records: list[PromptRecord] = []
    gateway.add_prompt_hook(records.append)
    result = await gateway.complete_result(TaskType.REPORT, "the prompt", system="sys")
    assert [(r.prompt_id, r.prompt, r.system, r.provider) for r in records] == [(result.prompt_id, "the prompt", "sys", "stub")]


def test_routing_file_is_loaded_and_validated(tmp_path, monkeypatch):
    path = tmp_path / "llm.yaml"
    path.write_text("tiers:\n  sql: [ollama:qwen2.5-coder, 'openai:gpt-4o']\n")
    monkeypatch.setenv("MLPILOT_LLM_CONFIG", str(path))
    routing = load_routing()
    assert routing["sql"] == [("ollama", "qwen2.5-coder"), ("openai", "gpt-4o")]
    assert routing["cheap"] == [tuple(e.split(":", 1)) for e in DEFAULT_TIERS["cheap"]]

    path.write_text("tiers:\n  cheap: [madeup:model]\n")
    with pytest.raises(ValueError, match="unknown provider 'madeup'"):
        load_routing()
    monkeypatch.setenv("MLPILOT_LLM_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_routing()


def test_example_config_is_valid():
    from pathlib import Path
    example = Path(__file__).resolve().parents[3] / "config" / "llm.example.yaml"
    assert set(load_routing(example)) == {"cheap", "reasoning", "sql"}
