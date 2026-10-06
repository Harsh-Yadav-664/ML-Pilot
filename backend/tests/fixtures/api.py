"""Helpers for tests that drive the HTTP API the way the UI does."""

from __future__ import annotations

from typing import Any

import httpx

API = "/api/v1"


async def create_project(client: httpx.AsyncClient, name: str = "Test project") -> str:
    resp = await client.post(
        f"{API}/projects/", json={"name": name, "task_type": "binary_classification"}
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def load_sample(
    client: httpx.AsyncClient, project_id: str, name: str = "telecom_churn"
) -> dict[str, Any]:
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/sample", json={"dataset_name": name}
    )
    assert resp.status_code == 200, resp.text
    return dict(resp.json())
