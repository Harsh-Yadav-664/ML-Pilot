"""Batch scoring of a finished run on the live database (#62).

The run is trained from the demo shop; the connection is then pointed at a DuckDB file holding
the same tables, which is what scoring reads: features are recomputed there with the stored SQL
through the SQL guard, and the stored model ranks the entities.
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest

import app.db.session as db_session
import tests.integration.test_baseline_run_api as base
from app.core import datasets
from app.db.models import Connection, Run
from ml.export.artifacts import REFERENCE_FILE, artifact_dir
from tests.integration.test_export_bundle import exported_run
from tests.integration.test_run_loop_api import runs_url

connection = base.connection  # the fixture, re-exported


async def scoring_setup(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, Path, Path]:
    _, db_path, run_id = await exported_run(client, project_id, connection, monkeypatch, tmp_path)
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        conn = await db.get(Connection, connection)
        assert conn is not None
        conn.dialect, conn.database = "duckdb", str(db_path)
        await db.commit()
    folder = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id)
    return run_id, db_path, folder


async def test_scoring_a_validation_cutoff_reproduces_the_stored_scores(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id, _, folder = await scoring_setup(client, project_id, connection, monkeypatch, tmp_path)
    reference = pd.read_csv(folder / REFERENCE_FILE)
    cutoff = str(reference["cutoff_time"].iloc[0])
    expected = reference[reference["cutoff_time"] == cutoff].set_index("entity_id")["score"]

    resp = await client.post(
        runs_url(project_id, f"/{run_id}/score"),
        json={"cutoff": cutoff.replace(" ", "T"), "top_k": 50},
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["summary"]["n_scored"] >= len(expected) > 0
    assert out["preview"][0]["rank"] == 1 and out["preview"][0]["decile"] == 1
    assert out["preview"][0]["reason_1"], out["preview"][0]
    scores = [p["score"] for p in out["preview"]]
    assert scores == sorted(scores, reverse=True)

    got = await client.get(runs_url(project_id, out["csv"].removeprefix("/runs")))
    assert got.status_code == 200 and got.headers["content-type"].startswith("text/csv")
    table = pd.read_csv(io.BytesIO(got.content)).set_index("entity_id")
    assert {"score", "rank", "decile", "reason_1", "reason_2", "reason_3"} <= set(table.columns)
    common = expected.index.intersection(table.index)
    assert len(common) == len(expected), "every validation entity must be scored"
    worst = float((table.loc[common, "score"] - expected.loc[common]).abs().max())
    print(f"max |score - stored validation score| over {len(common)} entities: {worst:.3g}")
    assert worst < 1e-9

    # the list is reachable again later, and an unknown stamp is a 404
    assert (
        await client.get(runs_url(project_id, f"/{run_id}/scores/19990101T000000.csv"))
    ).status_code == 404


async def test_the_default_cutoff_is_the_last_complete_day(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id, _, _ = await scoring_setup(client, project_id, connection, monkeypatch, tmp_path)
    resp = await client.post(runs_url(project_id, f"/{run_id}/score"), json={})
    assert resp.status_code == 200, resp.text
    cutoff = pd.Timestamp(resp.json()["cutoff"])
    assert cutoff == cutoff.floor("D")
    assert resp.json()["summary"]["n_scored"] > 0


async def test_a_dropped_column_stops_scoring_with_a_message_naming_it(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id, db_path, folder = await scoring_setup(
        client, project_id, connection, monkeypatch, tmp_path
    )
    cutoff = str(pd.read_csv(folder / REFERENCE_FILE)["cutoff_time"].iloc[0]).replace(" ", "T")
    ok = await client.post(runs_url(project_id, f"/{run_id}/score"), json={"cutoff": cutoff})
    assert ok.status_code == 200, ok.text

    con = duckdb.connect(str(db_path))
    con.execute('ALTER TABLE "orders" DROP COLUMN "total"')
    con.close()
    resp = await client.post(runs_url(project_id, f"/{run_id}/score"), json={"cutoff": cutoff})
    print(resp.status_code, resp.text)
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("Schema changed since training:") and "orders.total" in detail


async def test_scoring_an_unfinished_run_is_refused(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    assert (await client.post(runs_url(project_id, "/nope/score"), json={})).status_code == 404
