"""API refuses files outside MLPilot's data dirs and never echoes DB passwords."""

from __future__ import annotations

import httpx
import pytest

from app.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


BAD_PATHS = [
    "../../etc/passwd",
    "/etc/passwd",
    "datasets/../../../etc/passwd",
    "datasets/telecom_churn.py",
]


@pytest.mark.parametrize("bad", BAD_PATHS)
@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/ui/data/metrics",
        "/api/v1/ui/data/columns",
        "/api/v1/ui/data/leakage-warnings",
        "/api/v1/ui/agent/suggestions",
    ],
)
async def test_get_endpoints_reject_paths_outside_data_dirs(client, endpoint, bad):
    resp = await client.get(endpoint, params={"dataset_path": bad, "target_column": "Churn"})
    assert resp.status_code == 400
    assert "root:" not in resp.text


@pytest.mark.parametrize("bad", BAD_PATHS)
@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/ui/experiments/baseline",
        "/api/v1/ui/experiments/run",
        "/api/v1/ui/agent/auto-clean",
        "/api/v1/ui/agent/auto-optimize",
    ],
)
async def test_post_endpoints_reject_paths_outside_data_dirs(client, endpoint, bad):
    resp = await client.post(endpoint, json={"dataset_path": bad, "target_column": "Churn"})
    assert resp.status_code == 400


async def test_create_experiment_rejects_bad_dataset_version(client):
    resp = await client.post(
        "/api/v1/experiments/",
        json={
            "project_id": "p",
            "dataset_version": "../../etc/passwd",
            "hypothesis": "h",
            "change_description": "c",
            "model_name": "RandomForestClassifier",
            "parameters": {"target_column": "x"},
        },
    )
    assert resp.status_code == 400


async def test_sample_dataset_is_allowed(client):
    resp = await client.post("/api/v1/ui/data/sample", json={})
    assert resp.status_code == 200
    path = resp.json()["dataset_path"]
    resp = await client.get(
        "/api/v1/ui/data/metrics", params={"dataset_path": path, "target_column": "Churn"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "conn",
    [
        "postgresql://alice:s3cretpw@127.0.0.1:1/sales",
        "notadialect://alice:s3cretpw@127.0.0.1/sales",
        "sqlite:////nonexistent/dir/s3cretpw.db",
    ],
)
async def test_connect_sql_error_does_not_leak_password(client, conn):
    resp = await client.post(
        "/api/v1/ui/data/connect-sql", json={"connection_string": conn, "query": "SELECT 1"}
    )
    assert resp.status_code == 400
    assert "s3cretpw" not in resp.text


async def test_connect_sql_refuses_writes(client, tmp_path):
    resp = await client.post(
        "/api/v1/ui/data/connect-sql",
        json={
            "connection_string": f"sqlite:///{tmp_path / 'x.db'}",
            "query": "SELECT 1; DROP TABLE t",
        },
    )
    assert resp.status_code == 400
    assert "Exactly one statement" in resp.json()["detail"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/ui/experiments/baseline",
        "/api/v1/ui/experiments/run",
        "/api/v1/ui/agent/auto-clean",
        "/api/v1/ui/agent/auto-optimize",
    ],
)
async def test_post_endpoints_require_dataset_path(client, endpoint):
    resp = await client.post(endpoint, json={"target_column": "Churn"})
    assert resp.status_code == 400
    assert "dataset_path is required" in resp.text


@pytest.mark.parametrize(
    "endpoint",
    ["/api/v1/ui/data/metrics", "/api/v1/ui/data/leakage-warnings", "/api/v1/ui/agent/suggestions"],
)
async def test_get_endpoints_require_dataset_path(client, endpoint):
    resp = await client.get(endpoint, params={"target_column": "Churn"})
    assert resp.status_code == 422
