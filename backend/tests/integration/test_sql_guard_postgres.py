"""The SQL guard (#44) against a real Postgres (the CI service container).

* Layer 1: every hostile query in tests/fixtures/hostile_sql.yaml is refused, with its code.
* Layer 2: with the parser switched off, the read-only transaction still refuses writes.
* Layer 3: the connection test reports ``can_write`` for a role that may INSERT.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pg8000.native
import pytest
import yaml
from cryptography.fernet import Fernet

from app.core import secrets
from ml.data import sql_guard
from ml.data.engine import QueryTimeout
from ml.data.sources import ConnectionFailure, ConnectionSpec, open_source
from ml.data.sql_guard import SqlRejected
from tests.fixtures.api import API
from tests.fixtures.postgres import pg_spec

BACKEND = Path(__file__).resolve().parents[2]
HOSTILE = yaml.safe_load((BACKEND / "tests" / "fixtures" / "hostile_sql.yaml").read_text())
TABLE = "mlpilot_guard_orders"
INSERTER = "mlpilot_guard_inserter"
INSERTER_PASSWORD = "inserter-pass-5512"
READER = "mlpilot_guard_reader"
READER_PASSWORD = "reader-pass-8820"
PWNED = Path("/tmp/mlpilot_guard_pwned")


def _admin(spec: ConnectionSpec) -> pg8000.native.Connection:
    return pg8000.native.Connection(
        user=spec.username or "",
        password=spec.password,
        host=spec.host or "",
        port=spec.port or 5432,
        database=spec.database,
    )


@pytest.fixture
def pg() -> Iterator[ConnectionSpec]:
    """A table, a sequence, a role that may INSERT and a role that may only SELECT."""
    spec = pg_spec()
    admin = _admin(spec)
    admin.run(f"DROP TABLE IF EXISTS {TABLE}, mlpilot_guard_new, mlpilot_guard_copy")
    admin.run("DROP SEQUENCE IF EXISTS mlpilot_guard_seq")
    admin.run(f"CREATE TABLE {TABLE} (id int PRIMARY KEY, customer text, amount numeric)")
    admin.run(
        f"INSERT INTO {TABLE} SELECT i, 'c' || (i % 4), i * 2.5 FROM generate_series(1, 40) i"
    )
    admin.run("CREATE SEQUENCE mlpilot_guard_seq")
    for role, password, grant in (
        (INSERTER, INSERTER_PASSWORD, "SELECT, INSERT"),
        (READER, READER_PASSWORD, "SELECT"),
    ):
        if _role_exists(admin, role):
            admin.run(f"DROP OWNED BY {role}")
        admin.run(f"DROP ROLE IF EXISTS {role}")
        admin.run(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
        admin.run(f"GRANT {grant} ON {TABLE} TO {role}")
    PWNED.unlink(missing_ok=True)
    try:
        yield spec
    finally:
        admin.run(f"DROP TABLE IF EXISTS {TABLE}, mlpilot_guard_new, mlpilot_guard_copy")
        admin.run("DROP SEQUENCE IF EXISTS mlpilot_guard_seq")
        for role in (INSERTER, READER):
            admin.run(f"DROP OWNED BY {role}")
            admin.run(f"DROP ROLE {role}")
        admin.close()


def _role_exists(admin: pg8000.native.Connection, role: str) -> bool:
    return bool(admin.run("SELECT 1 FROM pg_roles WHERE rolname = :r", r=role))


def _state(spec: ConnectionSpec) -> tuple[Any, ...]:
    """Everything a write could change: the rows, the table list, the sequence."""
    admin = _admin(spec)
    try:
        rows = admin.run(f"SELECT id, customer, amount FROM {TABLE} ORDER BY id")
        tables = admin.run(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'mlpilot_guard%' ORDER BY 1"
        )
        columns = admin.run(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t", t=TABLE
        )
        seq = admin.run("SELECT last_value, is_called FROM mlpilot_guard_seq")
        return (rows, tables, columns, seq)
    finally:
        admin.close()


def test_the_hostile_matrix_is_refused_by_postgres(pg: ConnectionSpec) -> None:
    src = open_source(pg)
    before = _state(pg)
    refused = []
    for case in HOSTILE["postgres"]:
        sql = case["sql"].replace("{t}", TABLE)
        with pytest.raises(SqlRejected) as err:
            src.query(sql, limit=100, timeout_s=5)
        assert err.value.code == case["code"], (sql, err.value.reason)
        refused.append((case["code"], sql))
    for sql in HOSTILE["allowed"]["postgres"]:
        assert src.query(sql.replace("{t}", TABLE), limit=1000, timeout_s=10).num_rows > 0
    assert _state(pg) == before
    assert not PWNED.exists()
    print(f"\n{len(refused)} hostile queries refused by the guard against Postgres:")
    for code, sql in refused:
        print(f"  {code:20} {sql}")
    assert len(refused) >= 15


# Single statements a parser bug could let through. Each must fail in the session itself.
WRITES_WITHOUT_LAYER_1 = [
    f"INSERT INTO {TABLE} VALUES (99, 'x', 1)",
    f"UPDATE {TABLE} SET amount = 0",
    f"DELETE FROM {TABLE}",
    f"TRUNCATE {TABLE}",
    f"DROP TABLE {TABLE}",
    "CREATE TABLE mlpilot_guard_new (a int)",
    f"ALTER TABLE {TABLE} ADD COLUMN extra int",
    f"SELECT * INTO mlpilot_guard_copy FROM {TABLE}",
    f"WITH d AS (DELETE FROM {TABLE} RETURNING *) SELECT count(*) FROM d",
    f"SELECT * FROM {TABLE} FOR UPDATE",
    "SELECT nextval('mlpilot_guard_seq')",
    "SELECT set_config('transaction_read_only', 'off', true)",
    f"DO $$ BEGIN DELETE FROM {TABLE}; END $$",
]


def _unguarded(src: Any, sql: str) -> Any:
    """Layer 1 switched off: the SQL goes straight into the read-only session."""
    with sql_guard.read_only_session(src, 5) as session:
        return session.fetch_rows(sql, [], max_rows=None, max_bytes=10**6)


@pytest.mark.parametrize("sql", WRITES_WITHOUT_LAYER_1)
def test_with_the_parser_off_postgres_still_refuses_writes(pg: ConnectionSpec, sql: str) -> None:
    src = open_source(pg)
    before = _state(pg)
    with pytest.raises(ConnectionFailure) as err:
        _unguarded(src, sql)
    # 25006 read_only_sql_transaction; 25001 for set_config, which may not change the
    # transaction mode once the transaction has started.
    assert re.search(r"error code 2500[16]", str(err.value)), err.value
    assert _state(pg) == before


def test_with_the_parser_off_postgres_refuses_a_second_statement(pg: ConnectionSpec) -> None:
    """A smuggled ``ROLLBACK; DELETE`` would leave the read-only transaction; the extended
    protocol pg8000 uses accepts one statement per call, so the server refuses it."""
    src = open_source(pg)
    before = _state(pg)
    for sql in (f"ROLLBACK; DELETE FROM {TABLE}", f"COMMIT; DROP TABLE {TABLE}"):
        with pytest.raises(ConnectionFailure) as err:
            _unguarded(src, sql)
        assert "42601" in str(err.value), err.value  # syntax_error: multiple commands
    assert _state(pg) == before


def test_with_the_parser_off_the_session_cannot_be_switched_to_read_write(
    pg: ConnectionSpec,
) -> None:
    src = open_source(pg)
    with sql_guard.read_only_session(src, 5) as session:
        session.fetch_rows("SELECT 1", [], max_rows=None, max_bytes=10**6)  # takes a snapshot
        with pytest.raises(ConnectionFailure) as err:
            session.fetch_rows("SET TRANSACTION READ WRITE", [], max_rows=None, max_bytes=10**6)
        assert "25001" in str(err.value), err.value  # active_sql_transaction


def test_statement_and_idle_timeouts_are_set_for_every_session(pg: ConnectionSpec) -> None:
    src = open_source(pg)
    with sql_guard.read_only_session(src, 3) as session:
        _, rows = session.fetch_rows(
            "SELECT current_setting('statement_timeout'), "
            "current_setting('idle_in_transaction_session_timeout'), "
            "current_setting('transaction_read_only')",
            [],
            max_rows=None,
            max_bytes=10**6,
        )
    assert rows == [("3s", "8s", "on")]
    with pytest.raises(QueryTimeout):
        src.query("SELECT count(*) FROM generate_series(1, 10000000000) AS g", limit=1, timeout_s=1)


def test_a_reader_role_gets_the_same_answers_through_the_guard(pg: ConnectionSpec) -> None:
    reader = open_source(dataclasses.replace(pg, username=READER, password=READER_PASSWORD))
    table = reader.query(f"SELECT count(*) AS n FROM {TABLE}", limit=1, timeout_s=5)
    assert table.column("n").to_pylist() == [40]
    with pytest.raises(ConnectionFailure, match="42P01"):  # undefined_table, short message
        reader.query("SELECT * FROM mlpilot_guard_not_there", limit=1, timeout_s=5)


@pytest.fixture
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(secrets.KEY_VAR, Fernet.generate_key().decode())


@pytest.mark.usefixtures("secret_key")
async def test_the_test_endpoint_reports_can_write_for_a_role_with_insert(
    client: httpx.AsyncClient, project_id: str, pg: ConnectionSpec
) -> None:
    url = f"{API}/projects/{project_id}/connections/"
    results = {}
    for name, user, password in (
        ("inserter", INSERTER, INSERTER_PASSWORD),
        ("reader", READER, READER_PASSWORD),
    ):
        created = await client.post(
            url,
            json={
                "name": name,
                "dialect": "postgres",
                "host": pg.host,
                "port": pg.port,
                "database": pg.database,
                "username": user,
                "password": password,
            },
        )
        assert created.status_code == 201, created.text
        tested = await client.post(f"{url}{created.json()['id']}/test")
        assert tested.status_code == 200, tested.text
        results[name] = tested.json()
        stored = await client.get(f"{url}{created.json()['id']}")
        assert stored.json()["can_write"] == results[name]["can_write"]
    assert results["inserter"]["ok"] and results["inserter"]["can_write"] is True
    assert any(re.search(r"write to 1 table", n) for n in results["inserter"]["privilege_notes"])
    assert results["reader"]["ok"] and results["reader"]["can_write"] is False
    print(
        f"\ninserter: can_write={results['inserter']['can_write']} "
        f"notes={results['inserter']['privilege_notes']}"
    )
    print(
        f"reader:   can_write={results['reader']['can_write']} "
        f"notes={results['reader']['privilege_notes']}"
    )
