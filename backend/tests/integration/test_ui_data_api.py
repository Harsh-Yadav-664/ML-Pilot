"""Dataset endpoints return real stats or fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import datasets
from tests.fixtures.api import API, load_sample


async def test_columns_profile_sample(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    resp = await client.get(
        f"{API}/projects/{project_id}/datasets/{version}/columns",
        params={"target_column": "Churn"},
    )
    assert resp.status_code == 200, resp.text
    cols = {c["name"]: c for c in resp.json()}
    assert len(cols) == 21
    assert cols["Churn"]["role"] == "target"
    assert cols["customerID"]["role"] == "id"
    assert cols["tenure"]["dtype"] == "int"
    assert cols["Contract"]["dtype"] == "category" and cols["Contract"]["unique"] == 3
    for c in cols.values():
        assert len(c["dist"]) == 12 and max(c["dist"]) <= 1.0
        assert 0 <= c["missing_pct"] <= 100


@pytest.mark.parametrize("endpoint", ["metrics", "leakage", "columns"])
async def test_missing_version_is_an_error_not_zeros(client, project_id, endpoint):
    resp = await client.get(
        f"{API}/projects/{project_id}/datasets/{'f' * 64}/{endpoint}",
        params={"target_column": "Churn"},
    )
    assert resp.status_code == 404


async def test_leakage_on_a_missing_target_is_400(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    resp = await client.get(
        f"{API}/projects/{project_id}/datasets/{version}/leakage",
        params={"target_column": "nope"},
    )
    assert resp.status_code == 400


async def test_leakage_warnings_carry_category(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    resp = await client.get(
        f"{API}/projects/{project_id}/datasets/{version}/leakage",
        params={"target_column": "Churn"},
    )
    assert resp.status_code == 200, resp.text
    assert all("category" in w for w in resp.json())


async def test_run_rejects_unknown_model(client, project_id):
    version = (await load_sample(client, project_id))["data_version_id"]
    resp = await client.post(
        f"{API}/projects/{project_id}/experiments",
        json={
            "data_version_id": version,
            "target_column": "Churn",
            "model_name": "CatBoostClassifier",
        },
    )
    assert resp.status_code == 400
    assert "Unsupported model" in resp.json()["detail"]


async def test_upload_csv_returns_columns_and_rows(
    client, project_id, tmp_path, monkeypatch, isolated_storage
):
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path)
    csv = b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n"
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/upload",
        files={"file": ("../../evil.csv", csv, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns"] == ["a", "b", "label"]
    assert body["total_rows"] == 3
    assert body["filename"] == "evil.csv"
    assert "dataset_path" not in body  # clients address data by version id only
    # The crafted name can't escape: the stored copy is named by its content hash.
    assert [p.name for p in isolated_storage.iterdir()] == [f"{body['data_version_id']}.csv"]
    assert not list(tmp_path.glob("*evil*"))  # the raw upload isn't kept
    assert not list(Path(tmp_path).parent.glob("evil*"))


async def test_upload_rejects_non_csv(client, project_id):
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/upload",
        files={"file": ("data.xlsx", b"x", "application/octet-stream")},
    )
    assert resp.status_code == 400
