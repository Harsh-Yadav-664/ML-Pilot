"""API tests for AI feature suggestions, using the offline stub provider."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.api.deps import get_gateway
from app.main import app
from tests.fixtures.gateway import stub_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
async def client():
    app.dependency_overrides[get_gateway] = stub_gateway
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_ui_suggestions_on_sample_dataset(client):
    resp = await client.get(
        "/api/v1/ui/agent/suggestions",
        params={"dataset_path": str(SAMPLE), "target_column": "Churn"},
    )
    assert resp.status_code == 200, resp.text
    suggestions = resp.json()
    assert len(suggestions) >= 1
    assert all("name" in s and "formula" in s for s in suggestions)


async def test_experiments_suggest_on_sample_dataset(client):
    resp = await client.post(
        "/api/v1/experiments/suggest",
        json={
            "dataset_version": str(SAMPLE),
            "target_column": "Churn",
            "objective": "Predict churn",
            "max_hypotheses": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["hypotheses"]) == 2


async def test_ui_suggestions_missing_dataset_is_400(client):
    resp = await client.get(
        "/api/v1/ui/agent/suggestions",
        params={"dataset_path": "does/not/exist.csv", "target_column": "Churn"},
    )
    assert resp.status_code == 400


async def test_ui_suggestions_llm_failure_is_502(client, monkeypatch):
    gateway = stub_gateway()

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(gateway.providers["stub"], "complete_structured", boom)
    app.dependency_overrides[get_gateway] = lambda: gateway

    resp = await client.get(
        "/api/v1/ui/agent/suggestions",
        params={"dataset_path": str(SAMPLE), "target_column": "Churn"},
    )
    assert resp.status_code == 502
    assert "AI suggestions failed" in resp.json()["detail"]
