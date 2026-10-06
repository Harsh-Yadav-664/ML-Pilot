"""Immutable dataset versions: content-addressed, read-only, and what experiments train on."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.v1 import ui
from app.main import app
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.versions import content_hash, store_version, version_id_of
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _rows(db_url: str, version_id: str) -> list[tuple]:
    with sqlite3.connect(db_url.removeprefix("sqlite+aiosqlite:///")) as conn:
        return conn.execute(
            "SELECT kind, source, n_rows, n_columns FROM data_versions WHERE id = ?", (version_id,)
        ).fetchall()


def _toy_csv(path: Path) -> Path:
    X, y = make_classification(n_samples=400, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)
    df.to_csv(path, index=False)
    return path


async def _f1(dataset_path: str) -> float:
    spec = ExperimentSpec(
        id="v",
        project_id="p",
        dataset_version=dataset_path,
        hypothesis="h",
        change_description="c",
        model_name="LogisticRegression",
        parameters={"target_column": "target", "ensemble": False},
        feature_set=[],
    )
    executor = LocalExperimentExecutor(data_loader_func=lambda p: CsvLoader().load(p))
    return (await executor.run(spec)).metrics["f1"]


async def test_uploading_the_same_bytes_twice_yields_one_version(
    client, isolated_storage, metadata_db_url
):
    csv = b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n"
    first = (await client.post("/api/v1/ui/data/upload", files={"file": ("x.csv", csv)})).json()
    second = (await client.post("/api/v1/ui/data/upload", files={"file": ("y.csv", csv)})).json()
    assert first["data_version_id"] == second["data_version_id"] == hashlib.sha256(csv).hexdigest()
    assert first["dataset_path"] == second["dataset_path"]
    assert first["short_hash"] == first["data_version_id"][:12]
    assert len(list(isolated_storage.iterdir())) == 1
    ((kind, source, n_rows, n_cols),) = _rows(metadata_db_url, first["data_version_id"])
    assert (kind, n_rows, n_cols) == ("file", 3, 3)
    assert '"upload"' in source and '"x.csv"' in source  # first upload's name is kept


async def test_modifying_the_original_after_loading_does_not_change_a_rerun(
    client, tmp_path, monkeypatch
):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    original = _toy_csv(datasets_dir / "toy.csv")
    monkeypatch.setattr(ui, "DATASETS_DIR", datasets_dir)

    loaded = (await client.post("/api/v1/ui/data/sample", json={"dataset_name": "toy"})).json()
    path = loaded["dataset_path"]
    before = await _f1(path)

    # Overwrite the original with pure-noise features. The stored version is untouched.
    df = pd.read_csv(original)
    noise = np.random.default_rng(1).normal(size=(len(df), 5))
    df[[f"f{i}" for i in range(5)]] = noise
    df.to_csv(original, index=False)
    assert await _f1(path) == before
    assert await _f1(str(original)) != before  # proves the edit would have mattered
    assert content_hash(Path(path)) == loaded["data_version_id"]

    # Loading the changed file makes a new version; the old one is still there.
    reloaded = (await client.post("/api/v1/ui/data/sample", json={"dataset_name": "toy"})).json()
    assert reloaded["data_version_id"] != loaded["data_version_id"]
    assert Path(path).exists()


def test_stored_version_is_read_only_and_named_by_hash(tmp_path):
    src = _toy_csv(tmp_path / "Data.CSV")
    stored = store_version(src, tmp_path / "versions")
    assert stored.created and stored.path.name == f"{content_hash(src)}.csv"
    assert stat.S_IMODE(stored.path.stat().st_mode) == 0o444
    again = store_version(src, tmp_path / "versions")
    assert not again.created and again.path == stored.path
    assert sorted(p.name for p in (tmp_path / "versions").iterdir()) == [stored.path.name]


def test_version_id_is_only_read_from_the_versions_dir(tmp_path):
    vdir = tmp_path / "versions"
    digest = "a" * 64
    assert version_id_of(vdir / f"{digest}.csv", vdir) == digest
    assert version_id_of(tmp_path / f"{digest}.csv", vdir) is None
    assert version_id_of(vdir / "short.csv", vdir) is None


async def test_experiments_record_their_data_version(isolated_storage, metadata_db_url):
    stored = store_version(_toy_csv(isolated_storage.parent / "t.csv"), isolated_storage)
    factory = async_sessionmaker(create_async_engine(metadata_db_url, poolclass=NullPool))
    async with factory() as db:
        exp = await ExperimentService(db).create(
            ExperimentCreate(
                project_id="p",
                dataset_version=str(stored.path),
                hypothesis="h",
                change_description="c",
                model_name="LogisticRegression",
            )
        )
        await db.rollback()
    assert exp.data_version_id == stored.id
