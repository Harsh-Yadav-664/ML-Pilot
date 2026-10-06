"""Live sources (#43): Postgres, SQLite and DuckDB files are read-only, limited and well-behaved."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pg8000.native
import pytest

from ml.data import sql_guard
from ml.data.engine import EngineError, QueryRejected, QueryTimeout, RowLimitExceeded, TableRef
from ml.data.sources import ConnectionFailure, ConnectionSpec, open_source
from ml.data.sources.postgres import PostgresSource
from tests.fixtures.postgres import pg_spec

RO_ROLE = "mlpilot_test_reader"
RO_PASSWORD = "reader-pass-4471"


@pytest.fixture
def pg() -> Iterator[ConnectionSpec]:
    """The test Postgres with a small table and a read-only role, cleaned up afterwards."""
    spec = pg_spec()
    admin = pg8000.native.Connection(
        user=spec.username or "",
        password=spec.password,
        host=spec.host or "",
        port=spec.port or 5432,
        database=spec.database,
    )
    admin.run("DROP TABLE IF EXISTS mlpilot_test_orders")
    admin.run(
        "CREATE TABLE mlpilot_test_orders (id int PRIMARY KEY, customer text, amount numeric)"
    )
    admin.run(
        "INSERT INTO mlpilot_test_orders SELECT i, 'c' || (i % 5), i * 1.5 FROM generate_series(1, 20) i"
    )
    admin.run(f"DROP ROLE IF EXISTS {RO_ROLE}")
    admin.run(f"CREATE ROLE {RO_ROLE} LOGIN PASSWORD '{RO_PASSWORD}'")
    admin.run(f"GRANT SELECT ON mlpilot_test_orders TO {RO_ROLE}")
    try:
        yield spec
    finally:
        admin.run("DROP TABLE IF EXISTS mlpilot_test_orders")
        admin.run(f"DROP OWNED BY {RO_ROLE}")
        admin.run(f"DROP ROLE {RO_ROLE}")
        admin.close()


def test_postgres_lists_describes_and_queries_tables(pg: ConnectionSpec) -> None:
    src = open_source(pg)
    src.check()
    assert src.server_version()[0].isdigit()
    table = TableRef("mlpilot_test_orders", "public")
    assert table in src.list_tables()
    schema = src.table_schema(table)
    assert [(c.name, c.type) for c in schema.columns] == [
        ("id", "integer"),
        ("customer", "text"),
        ("amount", "numeric"),
    ]
    assert schema.primary_key == ["id"]
    result = src.query(
        "SELECT customer, count(*) AS n FROM mlpilot_test_orders GROUP BY customer ORDER BY 1",
        limit=100,
        timeout_s=10,
    )
    assert result.num_rows == 5 and set(result.column("n").to_pylist()) == {4}
    assert len(src.fingerprint()) == 64


def test_postgres_enforces_select_only_row_limit_and_timeout(pg: ConnectionSpec) -> None:
    src = open_source(pg)
    for bad in (
        "DROP TABLE mlpilot_test_orders",
        "DELETE FROM mlpilot_test_orders",
        "SELECT 1; DROP TABLE mlpilot_test_orders",
        "SELECT * INTO new_orders FROM mlpilot_test_orders",
    ):
        with pytest.raises(QueryRejected):
            src.query(bad, limit=10, timeout_s=5)
    with pytest.raises(RowLimitExceeded):
        src.query("SELECT * FROM mlpilot_test_orders", limit=5, timeout_s=5)
    with pytest.raises(QueryTimeout):
        src.query("SELECT count(*) FROM generate_series(1, 10000000000) AS g", limit=5, timeout_s=1)
    assert src.query("SELECT count(*) AS n FROM mlpilot_test_orders", limit=1, timeout_s=5).column(
        "n"
    ).to_pylist() == [20]  # nothing was dropped or deleted


def test_postgres_refuses_writes_even_if_the_sql_guard_were_bypassed(pg: ConnectionSpec) -> None:
    """Layer 2: the transaction itself is READ ONLY, whatever SQL reaches it."""
    src = PostgresSource(pg)
    for sql in ("CREATE TABLE should_not_exist (a int)", "DELETE FROM mlpilot_test_orders"):
        with (
            pytest.raises(ConnectionFailure, match="25006"),  # read_only_sql_transaction
            sql_guard.read_only_session(src, 5) as session,
        ):
            session.fetch_rows(sql, [], max_rows=None, max_bytes=10**6)
    assert src.query("SELECT count(*) AS n FROM mlpilot_test_orders", limit=1, timeout_s=5).column(
        "n"
    ).to_pylist() == [20]


def test_postgres_privilege_check_tells_a_writer_from_a_reader(pg: ConnectionSpec) -> None:
    writer = open_source(pg).privileges()
    assert writer.can_write and writer.notes
    reader = open_source(dataclasses.replace(pg, username=RO_ROLE, password=RO_PASSWORD))
    privileges = reader.privileges()
    assert not privileges.can_write, privileges.notes
    assert (
        reader.query("SELECT count(*) AS n FROM mlpilot_test_orders", limit=1, timeout_s=5).num_rows
        == 1
    )


def test_postgres_errors_are_short_fixed_messages(pg: ConnectionSpec) -> None:
    cases = {
        "auth_failed": dataclasses.replace(pg, password="definitely-wrong-pw"),
        "database_not_found": dataclasses.replace(pg, database="no_such_database_here"),
        "host_unreachable": dataclasses.replace(pg, port=1),
        "ssl_failed": dataclasses.replace(pg, ssl_mode="require"),
    }
    for code, spec in cases.items():
        with pytest.raises(ConnectionFailure) as err:
            open_source(spec).check()
        assert err.value.code == code, (code, err.value.message)
        for secret in (spec.password or "", spec.host or "", spec.username or ""):
            assert secret not in err.value.message


def test_a_postgres_source_does_not_show_its_password_in_its_repr(pg: ConnectionSpec) -> None:
    assert (pg.password or "") not in repr(pg) and (pg.password or "") not in repr(open_source(pg))


# -- SQLite and DuckDB files -----------------------------------------------------------


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)")
    con.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)", [(i, f"c{i % 3}", i * 2.5) for i in range(1, 11)]
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def duckdb_file(tmp_path: Path) -> Path:
    path = tmp_path / "shop.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE orders AS SELECT i AS id, 'c' || (i % 3) AS customer, i * 2.5 AS amount FROM range(1, 11) t(i)"
    )
    con.close()
    return path


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_file_databases_list_describe_query_and_report_read_only(
    dialect: str, sqlite_file: Path, duckdb_file: Path
) -> None:
    path = sqlite_file if dialect == "sqlite" else duckdb_file
    src = open_source(ConnectionSpec(dialect, str(path)))  # type: ignore[arg-type]
    src.check()
    assert dialect in src.server_version().lower()
    assert [t.name for t in src.list_tables()] == ["orders"]
    schema = src.table_schema(TableRef("orders", "main"))
    assert [c.name for c in schema.columns] == ["id", "customer", "amount"]
    table = src.query(
        "SELECT customer, sum(amount) AS total FROM orders GROUP BY 1 ORDER BY 1",
        limit=10,
        timeout_s=5,
    )
    assert table.num_rows == 3
    assert not src.privileges().can_write
    assert len(src.fingerprint()) == 64


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_file_databases_refuse_writes_and_limits_apply(
    dialect: str, sqlite_file: Path, duckdb_file: Path
) -> None:
    path = sqlite_file if dialect == "sqlite" else duckdb_file
    src = open_source(ConnectionSpec(dialect, str(path)))  # type: ignore[arg-type]
    for bad in (
        "DROP TABLE orders",
        "DELETE FROM orders",
        "INSERT INTO orders VALUES (99, 'x', 1)",
    ):
        with pytest.raises(QueryRejected):
            src.query(bad, limit=10, timeout_s=5)
    with pytest.raises(RowLimitExceeded):
        src.query("SELECT * FROM orders", limit=3, timeout_s=5)
    assert src.query("SELECT count(*) AS n FROM orders", limit=1, timeout_s=5).column(
        "n"
    ).to_pylist() == [10]


def test_sqlite_is_opened_read_only_at_the_file_level(sqlite_file: Path) -> None:
    src = open_source(ConnectionSpec("sqlite", str(sqlite_file)))
    con = src.connect(5)  # the file handle alone, before the guard's session settings
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("INSERT INTO orders VALUES (99, 'x', 1)")
    con.close()


def test_sqlite_stops_a_runaway_query(sqlite_file: Path) -> None:
    src = open_source(ConnectionSpec("sqlite", str(sqlite_file)))
    with pytest.raises(QueryTimeout):
        src.query(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
            limit=1,
            timeout_s=1,
        )


def test_duckdb_file_cannot_read_other_files(duckdb_file: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    src = open_source(ConnectionSpec("duckdb", str(duckdb_file)))
    with pytest.raises(EngineError):
        src.query(f"SELECT * FROM read_csv('{secret}')", limit=10, timeout_s=5)


def test_a_file_of_the_wrong_kind_is_not_opened(tmp_path: Path, sqlite_file: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("not a database")
    for dialect, path in (("sqlite", text), ("duckdb", text), ("duckdb", sqlite_file)):
        with pytest.raises(ConnectionFailure) as err:
            open_source(ConnectionSpec(dialect, str(path))).check()  # type: ignore[arg-type]
        assert err.value.code == "not_a_database_file"
