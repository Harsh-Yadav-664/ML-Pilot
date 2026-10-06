"""API tests for AI feature suggestions, using the offline stub provider."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.api.deps import get_gateway
from app.main import app
from tests.fixtures.api import API, load_sample
from tests.fixtures.gateway import stub_gateway


@pytest.fixture(autouse=True)
def stub_llm() -> Iterator[None]:
    app.dependency_overrides[get_gateway] = stub_gateway
    yield
    app.dependency_overrides.pop(get_gateway, None)


async def _suggest(client, project_id: str, version: str):
    return await client.get(
        f"{API}/projects/{project_id}/datasets/{version}/suggestions",
        params={"target_column": "Churn"},
    )


async def test_suggestions_on_sample_dataset(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    resp = await _suggest(client, project_id, version)
    assert resp.status_code == 200, resp.text
    suggestions = resp.json()
    assert len(suggestions) >= 1
    assert all("name" in s and "formula" in s for s in suggestions)


async def test_suggestions_unknown_version_is_404(client, project_id):
    resp = await _suggest(client, project_id, "0" * 64)
    assert resp.status_code == 404


async def test_suggestions_llm_failure_is_502(client, project_id, monkeypatch):
    version = (await load_sample(client, project_id))["data_version_id"]
    gateway = stub_gateway()

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(gateway.providers["stub"], "complete_structured", boom)
    app.dependency_overrides[get_gateway] = lambda: gateway

    resp = await _suggest(client, project_id, version)
    assert resp.status_code == 502
    assert "AI suggestions failed" in resp.json()["detail"]
