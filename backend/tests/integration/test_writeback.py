"""Opt-in write-back of a run's scores to Postgres (#172).

Off by default: only a connection saved with writeback=true may write, and only to
mlpilot_predictions.scores. Needs the test Postgres (MLPILOT_TEST_PG_*), like the connection tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pandas as pd
import pg8000.native
import pytest

from ml.data.sources.postgres import PostgresSource
from ml.data.sql_guard import QueryRejected
from tests.fixtures.api import API
from tests.fixtures.postgres import pg_spec
from tests.integration.test_run_loop_api import runs_url
from tests.integration.test_run_score_api import connection, scoring_setup  # noqa: F401

OUTSIDE = """
    SELECT c.table_schema, c.table_name, c.column_name, c.data_type
    FROM information_schema.columns c
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema', 'mlpilot_predictions')
    ORDER BY 1, 2, 3
"""


def _outside(admin: pg8000.native.Connection) -> list[tuple[object, ...]]:
    """Every column outside mlpilot_predictions, and the row count of each table."""
    cols = [tuple(r) for r in admin.run(OUTSIDE)]
    tables = sorted({(s, t) for s, t, *_ in cols})
    counts = [(s, t, admin.run(f'SELECT count(*) FROM "{s}"."{t}"')[0][0]) for s, t in tables]
    return cols + counts


async def test_writeback_is_opt_in_and_only_touches_mlpilot_predictions(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    spec = pg_spec()
    password = spec.password or ""
    monkeypatch.setenv("WRITEBACK_TEST_PASSWORD", password)
    admin = pg8000.native.Connection(
        user=spec.username or "",
        password=password,
        host=spec.host or "",
        port=spec.port or 5432,
        database=spec.database,
    )
    admin.run("DROP SCHEMA IF EXISTS mlpilot_predictions CASCADE")
    exists = "SELECT count(*) FROM pg_namespace WHERE nspname = 'mlpilot_predictions'"
    run_id, _, folder = await scoring_setup(client, project_id, connection, monkeypatch, tmp_path)
    before = _outside(admin)
    body = {
        "dialect": "postgres",
        "host": spec.host,
        "port": spec.port,
        "database": spec.database,
        "username": spec.username,
        "password_env": "WRITEBACK_TEST_PASSWORD",
    }
    conns = f"{API}/projects/{project_id}/connections/"
    plain = await client.post(conns, json={**body, "name": "warehouse"})
    assert plain.status_code == 201 and plain.json()["writeback"] is False, plain.text
    url = runs_url(project_id, f"/{run_id}/writeback")
    responses = [plain]

    # Off by default: the read connection and a plain Postgres one both get 403, nothing written.
    for conn_id in (connection, plain.json()["id"]):
        resp = await client.post(url, json={"connection_id": conn_id})
        responses.append(resp)
        assert resp.status_code == 403, resp.text
    assert admin.run(exists)[0][0] == 0, "nothing may be written while write-back is off"

    created = await client.post(conns, json={**body, "name": "scores out", "writeback": True})
    responses.append(created)
    assert created.status_code == 201 and created.json()["writeback"] is True, created.text
    wb_id = created.json()["id"]
    sqlite = await client.post(
        conns, json={"name": "x", "dialect": "sqlite", "database": "x.db", "writeback": True}
    )
    assert sqlite.status_code == 400, "write-back is Postgres only"

    resp = await client.post(url, json={"connection_id": wb_id})
    responses.append(resp)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    print(out)
    assert out["table"] == "mlpilot_predictions.scores" and out["rows_written"] > 0
    stamp = pd.Timestamp(out["cutoff"]).strftime("%Y%m%dT%H%M%S")
    expected = pd.read_csv(folder / "scores" / f"{stamp}.csv", dtype={"entity_id": str})
    rows = admin.run(
        "SELECT entity_id, score, cutoff FROM mlpilot_predictions.scores WHERE run_id = :r",
        r=run_id,
    )
    got = pd.DataFrame(rows, columns=["entity_id", "score", "cutoff"]).set_index("entity_id")
    assert len(got) == len(expected) == out["rows_written"]
    diff = (got["score"] - expected.set_index("entity_id")["score"]).abs().max()
    assert diff < 1e-12, diff
    print(f"{len(got)} rows in mlpilot_predictions.scores, max |score - csv| = {diff:.3g}")

    # Nothing outside mlpilot_predictions was created or changed.
    assert _outside(admin) == before
    # A write-back connection is never read from; the read-only path still refuses writes.
    schema = await client.get(f"{conns}{wb_id}/schema")
    assert schema.status_code == 400, schema.text
    with pytest.raises(QueryRejected):
        PostgresSource(spec).query(
            "INSERT INTO mlpilot_predictions.scores VALUES ('x','y',now(),1,now())",
            limit=10,
            timeout_s=10,
        )
    assert admin.run("SELECT count(*) FROM mlpilot_predictions.scores")[0][0] == len(got)

    # The write connection's password never shows in a response or a log line.
    leaks = [r.text for r in responses if password in r.text]
    assert not leaks and password not in caplog.text
    admin.run("DROP SCHEMA mlpilot_predictions CASCADE")
    admin.close()
