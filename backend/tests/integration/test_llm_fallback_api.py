"""A forced parse failure shows up as decision_mode 'fallback' in the API response."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from ai.gateway import PROVIDER_FACTORIES, AIGateway
from ai.llm_config import validate_tiers
from ai.router import TaskRouter
from app.api.deps import get_gateway
from app.main import app

SAMPLE = "datasets/telecom_churn.csv"


async def test_suggestions_report_the_fallback(monkeypatch):
    settings = SimpleNamespace(**{**{s: None for s, _, _ in PROVIDER_FACTORIES.values()}, "OPENAI_API_KEY": "test-key-not-real"})
    entry = ["openai:gpt-4o"]
    gateway = AIGateway(settings, router=TaskRouter(validate_tiers({"cheap": entry, "reasoning": entry, "sql": entry})))

    async def bad_json(**kwargs):
        raise json.JSONDecodeError("Expecting value", "not json", 0)

    monkeypatch.setattr(gateway.providers["openai"], "complete_structured", bad_json)
    app.dependency_overrides[get_gateway] = lambda: gateway
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/ui/agent/suggestions", params={"dataset_path": SAMPLE, "target_column": "Churn"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200, resp.text
    ideas = resp.json()
    assert ideas and all(i["llm"]["decision_mode"] == "fallback" for i in ideas)
    assert ideas[0]["llm"]["provider"] == "stub" and "openai: Expecting value" in ideas[0]["llm"]["errors"][0]
