"""Saved connections through the API (#43): create, test, list, delete, snapshot; no secret leaks."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import duckdb
import httpx
import pg8000.native
import pytest
from cryptography.fernet import Fernet

from app.core import datasets, secrets
from tests.fixtures.api import API, create_project
from tests.fixtures.postgres import pg_spec

PG_TABLE = "mlpilot_api_orders"


@pytest.fixture(autouse=True)
def secret_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv(secrets.KEY_VAR, key)
    return key


@pytest.fixture
async def pg_body() -> AsyncGenerator[dict[str, object]]:
    """The body of a create-connection request for the test Postgres, with a small table."""
    spec = pg_spec()
    admin = pg8000.native.Connection(
        user=spec.username or "",
        password=spec.password,
        host=spec.host or "",
        port=spec.port or 5432,
        database=spec.database,
    )
    admin.run(f"DROP TABLE IF EXISTS {PG_TABLE}")
    admin.run(f"CREATE TABLE {PG_TABLE} (id int, plan text, churned int)")
    admin.run(
        f"INSERT INTO {PG_TABLE} SELECT i, 'p' || (i % 3), i % 2 FROM generate_series(1, 30) i"
    )
    yield {
        "name": "sales",
        "dialect": "postgres",
        "host": spec.host,
        "port": spec.port,
        "database": spec.database,
        "username": spec.username,
        "password": spec.password,
        "ssl_mode": "prefer",
    }
    admin.run(f"DROP TABLE IF EXISTS {PG_TABLE}")
    admin.close()


def _url(project_id: str, tail: str = "") -> str:
    return f"{API}/projects/{project_id}/connections/{tail}".rstrip("/") + ("/" if not tail else "")


def _metadata_rows(db_url: str) -> list[tuple]:
    with sqlite3.connect(db_url.removeprefix("sqlite+aiosqlite:///")) as conn:
        return conn.execute("SELECT name, secret_ref FROM connections").fetchall()


async def test_create_test_list_delete_a_postgres_connection(
    client: httpx.AsyncClient, project_id: str, pg_body: dict[str, object], metadata_db_url: str
) -> None:
    password = str(pg_body["password"])
    created = await client.post(_url(project_id), json=pg_body)
    assert created.status_code == 201, created.text
    conn = created.json()
    assert conn["secret"] == "stored (encrypted)" and conn["dialect"] == "postgres"
    assert password not in created.text

    # What is saved is ciphertext, not the password.
    ((_, ref),) = [r for r in _metadata_rows(metadata_db_url) if r[0] == "sales"]
    assert ref.startswith("enc:") and password not in ref

    tested = await client.post(_url(project_id, f"{conn['id']}/test"))
    assert tested.status_code == 200, tested.text
    body = tested.json()
    assert body["ok"] is True and body["server_version"][0].isdigit()
    assert isinstance(body["can_write"], bool) and body["latency_ms"] >= 0
    assert password not in tested.text

    listed = await client.get(_url(project_id))
    assert [c["name"] for c in listed.json()] == ["sales"]
    assert listed.json()[0]["can_write"] == body["can_write"]
    assert listed.json()[0]["last_tested_at"] is not None
    assert password not in listed.text and "enc:" not in listed.text
    assert password not in (await client.get(_url(project_id, conn["id"]))).text

    deleted = await client.delete(_url(project_id, conn["id"]))
    assert deleted.status_code == 204
    assert (await client.get(_url(project_id))).json() == []
    assert (await client.get(_url(project_id, conn["id"]))).status_code == 404
    assert not [r for r in _metadata_rows(metadata_db_url) if r[0] == "sales"]


async def test_a_wrong_password_is_reported_without_the_password(
    client: httpx.AsyncClient, project_id: str, pg_body: dict[str, object]
) -> None:
    wrong = {**pg_body, "password": "wrong-password-8841"}
    conn = (await client.post(_url(project_id), json=wrong)).json()
    result = (await client.post(_url(project_id, f"{conn['id']}/test"))).json()
    assert result["ok"] is False and result["error_code"] == "auth_failed"
    assert (
        "wrong-password-8841" not in json.dumps(result)
        and str(pg_body["host"]) not in result["message"]
    )


async def test_storing_a_password_without_the_secret_key_is_refused(
    client: httpx.AsyncClient,
    project_id: str,
    pg_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(secrets.KEY_VAR)
    resp = await client.post(_url(project_id), json=pg_body)
    assert resp.status_code == 400
    assert "MLPILOT_SECRET_KEY is not set" in resp.json()["detail"]
    assert str(pg_body["password"]) not in resp.text
    assert (await client.get(_url(project_id))).json() == []  # nothing was saved


async def test_a_password_can_come_from_an_environment_variable_instead(
    client: httpx.AsyncClient,
    project_id: str,
    pg_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(secrets.KEY_VAR)  # no key needed: nothing secret is stored
    monkeypatch.setenv("PGPASSWORD_SALES", str(pg_body.pop("password")))
    resp = await client.post(_url(project_id), json={**pg_body, "password_env": "PGPASSWORD_SALES"})
    assert resp.status_code == 201 and resp.json()["secret"] == "env:PGPASSWORD_SALES"
    result = await client.post(_url(project_id, f"{resp.json()['id']}/test"))
    assert result.json()["ok"] is True
    both = await client.post(
        _url(project_id),
        json={**pg_body, "name": "x", "password": "p", "password_env": "PGPASSWORD_SALES"},
    )
    assert both.status_code == 400


async def test_a_full_connect_test_query_cycle_at_debug_level_logs_no_secret(
    client: httpx.AsyncClient,
    project_id: str,
    pg_body: dict[str, object],
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    password = str(pg_body["password"])
    caplog.set_level(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    responses: list[str] = []

    created = await client.post(_url(project_id), json=pg_body)
    conn_id = created.json()["id"]
    wrong = await client.post(
        _url(project_id), json={**pg_body, "name": "bad", "password": "bad-password-3377"}
    )
    responses += [created.text, wrong.text]
    responses.append((await client.post(_url(project_id, f"{conn_id}/test"))).text)
    responses.append((await client.post(_url(project_id, f"{wrong.json()['id']}/test"))).text)
    snapshot = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_id": conn_id, "query": f"SELECT id, plan, churned FROM {PG_TABLE}"},
    )
    assert snapshot.status_code == 200, snapshot.text
    responses.append(snapshot.text)
    refused = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_id": conn_id, "query": f"DROP TABLE {PG_TABLE}"},
    )
    assert refused.status_code == 400
    responses.append(refused.text)

    out, err = capfd.readouterr()
    haystack = "\n".join(
        [caplog.text, out, err, *responses]
        + [r.exc_text or "" for r in caplog.records]
        + [r.getMessage() for r in caplog.records]
    )
    assert len(caplog.records) > 10, "the DEBUG cycle should have produced log records to scan"
    for secret in (password, "bad-password-3377"):
        assert secret not in haystack
    # No connection string either: no URL with credentials, no key=value password pairs.
    assert not re.search(r"://[^\s/:@]+:[^\s@*]+@", haystack)
    assert not re.search(r"password\s*[=:]\s*\S", haystack, re.IGNORECASE)


async def test_a_snapshot_through_a_saved_connection_becomes_a_data_version(
    client: httpx.AsyncClient, project_id: str, pg_body: dict[str, object]
) -> None:
    conn = (await client.post(_url(project_id), json=pg_body)).json()
    resp = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={
            "connection_id": conn["id"],
            "query": f"SELECT id, plan, churned FROM {PG_TABLE} ORDER BY id",
        },
    )
    assert resp.status_code == 200, resp.text
    info = resp.json()
    assert info["columns"] == ["id", "plan", "churned"] and info["total_rows"] == 30
    assert (datasets.VERSIONS_DIR / f"{info['data_version_id']}.csv").exists()


async def test_the_old_connection_string_flow_still_works_and_needs_exactly_one_source(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    db = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(i, "x") for i in range(5)])
    con.commit()
    con.close()
    url = f"{API}/projects/{project_id}/datasets/sql"
    ok = await client.post(
        url, json={"connection_string": f"sqlite:///{db}", "query": "SELECT * FROM t"}
    )
    assert ok.status_code == 200 and ok.json()["total_rows"] == 5
    neither = await client.post(url, json={"query": "SELECT 1"})
    assert neither.status_code == 400
    both = await client.post(
        url,
        json={"connection_string": f"sqlite:///{db}", "connection_id": "x", "query": "SELECT 1"},
    )
    assert both.status_code == 400


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
async def test_file_connections_create_test_and_snapshot(
    dialect: str, client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    path = tmp_path / f"shop.{dialect}"
    if dialect == "sqlite":
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE orders (id INTEGER, amount REAL)")
        con.executemany("INSERT INTO orders VALUES (?, ?)", [(i, i * 1.5) for i in range(8)])
        con.commit()
        con.close()
    else:
        d = duckdb.connect(str(path))
        d.execute("CREATE TABLE orders AS SELECT i AS id, i * 1.5 AS amount FROM range(8) t(i)")
        d.close()
    created = await client.post(
        _url(project_id), json={"name": "shop", "dialect": dialect, "database": str(path)}
    )
    assert created.status_code == 201, created.text
    assert created.json()["secret"] is None
    tested = (await client.post(_url(project_id, f"{created.json()['id']}/test"))).json()
    assert tested["ok"] is True and tested["can_write"] is False
    snap = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_id": created.json()["id"], "query": "SELECT id, amount FROM orders"},
    )
    assert snap.status_code == 200 and snap.json()["total_rows"] == 8
    write = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_id": created.json()["id"], "query": "DELETE FROM orders"},
    )
    assert write.status_code == 400 and "SELECT" in write.json()["detail"]


async def test_file_connections_only_open_real_database_files_outside_mlpilots_own_data(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    own = datasets.VERSIONS_DIR
    own.mkdir(parents=True, exist_ok=True)
    (own / "x.sqlite").write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    for path in (str(text), str(tmp_path / "missing.sqlite"), str(own / "x.sqlite")):
        resp = await client.post(
            _url(project_id), json={"name": path[-12:], "dialect": "sqlite", "database": path}
        )
        assert resp.status_code == 400, (path, resp.text)


async def test_names_are_unique_and_connections_belong_to_their_project(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    path = tmp_path / "a.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (a INTEGER)")
    con.commit()
    con.close()
    body = {"name": "one", "dialect": "sqlite", "database": str(path)}
    first = await client.post(_url(project_id), json=body)
    assert first.status_code == 201
    assert (await client.post(_url(project_id), json=body)).status_code == 409
    other = await create_project(client, "Other project")
    assert (await client.get(_url(other, first.json()["id"]))).status_code == 404
    assert (await client.post(_url(other, f"{first.json()['id']}/test"))).status_code == 404
    assert (await client.delete(_url(other, first.json()["id"]))).status_code == 404
    assert (await client.get(_url(other))).json() == []


async def test_patch_changes_fields_and_a_new_password_replaces_the_old_one(
    client: httpx.AsyncClient, project_id: str, pg_body: dict[str, object]
) -> None:
    wrong = await client.post(_url(project_id), json={**pg_body, "password": "wrong-password-1234"})
    conn_id = wrong.json()["id"]
    assert (await client.post(_url(project_id, f"{conn_id}/test"))).json()["ok"] is False
    patched = await client.patch(
        _url(project_id, conn_id), json={"name": "sales2", "password": str(pg_body["password"])}
    )
    assert patched.status_code == 200 and patched.json()["name"] == "sales2"
    assert patched.json()["last_tested_at"] is None  # what was tested is not what is saved now
    assert (await client.post(_url(project_id, f"{conn_id}/test"))).json()["ok"] is True


async def test_a_validation_error_does_not_echo_the_password_or_connection_string(
    client: httpx.AsyncClient, project_id: str
) -> None:
    """A missing field makes FastAPI report the whole body as the error's input."""
    missing_database = await client.post(
        _url(project_id),
        json={"name": "x", "dialect": "postgres", "host": "h", "password": "echo-me-not-5521"},
    )
    assert missing_database.status_code == 422
    assert "echo-me-not-5521" not in missing_database.text
    assert missing_database.json()["detail"][0]["loc"] == ["body", "database"]  # still useful

    missing_query = await client.post(
        f"{API}/projects/{project_id}/datasets/sql",
        json={"connection_string": "postgresql://u:echo-me-not-9917@h/db"},
    )
    assert missing_query.status_code == 422
    assert "echo-me-not-9917" not in missing_query.text
