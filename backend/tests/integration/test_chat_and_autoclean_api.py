"""API tests for chat Q&A, debrief and Auto-Clean, using the offline stub provider."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

from app.api.deps import get_gateway
from app.main import app
from tests.fixtures.api import API
from tests.fixtures.gateway import stub_gateway


@pytest.fixture(autouse=True)
def stub_llm() -> Iterator[None]:
    app.dependency_overrides[get_gateway] = stub_gateway
    yield
    app.dependency_overrides.pop(get_gateway, None)


async def _upload(client, project_id) -> str:
    csv = pd.DataFrame({"age": [20, 30, 40, 50] * 5, "target": [0, 1, 0, 1] * 5}).to_csv(
        index=False
    )
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/upload", files={"file": ("d.csv", csv.encode())}
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["data_version_id"])


async def test_chat_ask_answers_with_stub(client, project_id):
    resp = await client.post(
        f"{API}/projects/{project_id}/chat/ask", json={"query": "Which model was best?"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert "grounding_context" in body


async def test_debrief_for_existing_experiment(client, project_id):
    version = await _upload(client, project_id)
    exp = await client.post(
        f"{API}/projects/{project_id}/experiments/baseline",
        json={"data_version_id": version, "target_column": "target"},
    )
    assert exp.status_code == 200, exp.text
    resp = await client.get(f"{API}/projects/{project_id}/experiments/{exp.json()['id']}/debrief")
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json()["debrief"], str) and resp.json()["debrief"]


async def test_debrief_unknown_experiment_is_404(client, project_id):
    resp = await client.get(f"{API}/projects/{project_id}/experiments/does-not-exist/debrief")
    assert resp.status_code == 404


async def test_auto_clean_surfaces_llm_failure(client, project_id, monkeypatch):
    version = await _upload(client, project_id)

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    gateway = stub_gateway()
    monkeypatch.setattr(gateway.providers["stub"], "complete_structured", boom)
    app.dependency_overrides[get_gateway] = lambda: gateway

    resp = await client.post(
        f"{API}/projects/{project_id}/experiments/auto-clean",
        json={"data_version_id": version, "target_column": "target"},
    )
    assert resp.status_code == 502
    assert "AI cleaning strategy failed" in resp.json()["detail"]
