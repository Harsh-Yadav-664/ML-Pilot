"""A queued baseline must reach a final status, never stay 'created'."""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import pytest

from tests.fixtures.api import API

FINAL = {"completed", "failed"}


async def upload(client: httpx.AsyncClient, project_id: str, df: pd.DataFrame) -> str:
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/upload",
        files={"file": ("small.csv", df.to_csv(index=False).encode())},
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["data_version_id"])


@pytest.fixture
async def small_version(client, project_id) -> str:
    rows = 60
    df = pd.DataFrame(
        {
            "age": list(range(rows)),
            "plan": ["a", "b", "c"] * (rows // 3),
            "churn": ["Yes" if i % 3 == 0 else "No" for i in range(rows)],
        }
    )
    return await upload(client, project_id, df)


async def wait_for_final_status(
    client: httpx.AsyncClient, project_id: str, exp_id: str, timeout: float = 60
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        resp = await client.get(f"{API}/projects/{project_id}/experiments/{exp_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in FINAL or loop.time() > deadline:
            return body
        await asyncio.sleep(0.05)


async def test_baseline_reaches_completed_20_times(client, project_id, small_version):
    for _ in range(20):
        resp = await client.post(
            f"{API}/projects/{project_id}/experiments/baseline",
            json={"data_version_id": small_version, "target_column": "churn"},
        )
        assert resp.status_code == 200, resp.text
        body = await wait_for_final_status(client, project_id, resp.json()["id"])
        assert body["status"] == "completed", body.get("decision_reason")


async def test_baseline_failure_is_recorded_with_message(client, project_id, small_version):
    resp = await client.post(
        f"{API}/projects/{project_id}/experiments/baseline",
        json={"data_version_id": small_version, "target_column": "no_such_column"},
    )
    assert resp.status_code == 200, resp.text
    body = await wait_for_final_status(client, project_id, resp.json()["id"])
    assert body["status"] == "failed"
    assert body["decision_reason"]
