"""The API takes no file paths, stays inside projects and never echoes DB passwords."""

from __future__ import annotations

import pytest

from tests.fixtures.api import API, create_project, load_sample

BAD_IDS = [
    "../../etc/passwd",
    "..%2F..%2Fetc%2Fpasswd",
    "a" * 64,  # well-formed, but no such version
    "A" * 64,
    "telecom_churn",
]

DATA_GETS = ["metrics", "columns", "leakage", "suggestions"]
VERSION_POSTS = [
    "experiments",
    "experiments/baseline",
    "experiments/auto-clean",
    "agent/auto-optimize",
]


@pytest.mark.parametrize("bad", BAD_IDS)
@pytest.mark.parametrize("endpoint", DATA_GETS)
async def test_get_endpoints_404_on_unknown_versions(client, project_id, endpoint, bad):
    resp = await client.get(
        f"{API}/projects/{project_id}/datasets/{bad}/{endpoint}", params={"target_column": "Churn"}
    )
    assert resp.status_code == 404
    assert "root:" not in resp.text


@pytest.mark.parametrize("bad", BAD_IDS)
@pytest.mark.parametrize("endpoint", VERSION_POSTS)
async def test_post_endpoints_404_on_unknown_versions(client, project_id, endpoint, bad):
    resp = await client.post(
        f"{API}/projects/{project_id}/{endpoint}",
        json={"data_version_id": bad, "target_column": "Churn"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("endpoint", VERSION_POSTS)
async def test_post_endpoints_require_a_data_version(client, project_id, endpoint):
    resp = await client.post(
        f"{API}/projects/{project_id}/{endpoint}", json={"target_column": "Churn"}
    )
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path",
    ["experiments", "datasets/sample", "chat/ask", "jobs/x"],
)
async def test_unknown_project_is_404(client, path):
    method = client.post if path in ("datasets/sample", "chat/ask") else client.get
    kwargs = {"json": {"query": "q"}} if path == "chat/ask" else {}
    resp = await method(f"{API}/projects/no-such-project/{path}", **kwargs)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Project not found"


async def test_sample_name_cannot_escape_the_samples_dir(client, project_id):
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/sample", json={"dataset_name": "../app/main"}
    )
    assert resp.status_code == 404


async def test_experiments_of_another_project_are_not_visible(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    base = (
        await client.post(
            f"{API}/projects/{project_id}/experiments/baseline",
            json={"data_version_id": version, "target_column": "Churn"},
        )
    ).json()
    other = await create_project(client, "Other")
    assert (await client.get(f"{API}/projects/{other}/experiments")).json() == []
    for suffix in ("", "/export", "/debrief"):
        resp = await client.get(f"{API}/projects/{other}/experiments/{base['id']}{suffix}")
        assert resp.status_code == 404


@pytest.mark.parametrize(
    "conn",
    [
        "postgresql://alice:s3cretpw@127.0.0.1:1/sales",
        "notadialect://alice:s3cretpw@127.0.0.1/sales",
        "sqlite:////nonexistent/dir/s3cretpw.db",
    ],
)
async def test_sql_snapshot_error_does_not_leak_password(client, project_id, conn):
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_string": conn, "query": "SELECT 1"},
    )
    assert resp.status_code == 400
    assert "s3cretpw" not in resp.text


async def test_sql_snapshot_refuses_writes(client, project_id, tmp_path):
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={
            "connection_string": f"sqlite:///{tmp_path / 'x.db'}",
            "query": "SELECT 1; DROP TABLE t",
        },
    )
    assert resp.status_code == 400
    assert "Exactly one statement" in resp.json()["detail"]
