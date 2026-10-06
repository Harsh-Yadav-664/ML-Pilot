"""Data endpoints used by the UI return real stats or fail loudly."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.main import app

SAMPLE = str(Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv")


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_columns_profile_sample(client):
    resp = await client.get(
        "/api/v1/ui/data/columns", params={"dataset_path": SAMPLE, "target_column": "Churn"}
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


@pytest.mark.parametrize(
    "path",
    ["/api/v1/ui/data/metrics", "/api/v1/ui/data/leakage-warnings", "/api/v1/ui/data/columns"],
)
async def test_missing_dataset_is_an_error_not_zeros(client, path):
    resp = await client.get(
        path, params={"dataset_path": "does/not/exist.csv", "target_column": "Churn"}
    )
    assert resp.status_code == 400


async def test_leakage_warnings_carry_category(client):
    resp = await client.get(
        "/api/v1/ui/data/leakage-warnings",
        params={"dataset_path": SAMPLE, "target_column": "Churn"},
    )
    assert resp.status_code == 200, resp.text
    assert all("category" in w for w in resp.json())


async def test_run_rejects_unknown_model(client):
    resp = await client.post(
        "/api/v1/ui/experiments/run",
        json={
            "dataset_path": SAMPLE,
            "target_column": "Churn",
            "model_name": "CatBoostClassifier",
            "feature_suggestion": {},
        },
    )
    assert resp.status_code == 400
    assert "Unsupported model" in resp.json()["detail"]


async def test_upload_csv_returns_columns_and_rows(client, tmp_path, monkeypatch, isolated_storage):
    from app.api.v1 import ui
    from app.core import datasets

    monkeypatch.setattr(ui, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(datasets, "ALLOWED_DATA_DIRS", [tmp_path])
    csv = b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n"
    resp = await client.post(
        "/api/v1/ui/data/upload", files={"file": ("../../evil.csv", csv, "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns"] == ["a", "b", "label"]
    assert body["total_rows"] == 3
    assert body["filename"] == "evil.csv"
    # The crafted name can't escape: the stored copy is named by its content hash.
    assert Path(body["dataset_path"]).parent == isolated_storage
    assert Path(body["dataset_path"]).name == f"{body['data_version_id']}.csv"
    assert not list(tmp_path.glob("*evil*"))  # the raw upload isn't kept


async def test_upload_rejects_non_csv(client):
    resp = await client.post(
        "/api/v1/ui/data/upload", files={"file": ("data.xlsx", b"x", "application/octet-stream")}
    )
    assert resp.status_code == 400
