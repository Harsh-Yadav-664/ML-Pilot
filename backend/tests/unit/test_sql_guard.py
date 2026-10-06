"""The SQL guard (#44) on SQLite and DuckDB files, and its parse layer for every dialect.

The Postgres half of the matrix is in tests/integration/test_sql_guard_postgres.py.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from ml.data import sql_guard
from ml.data.engine import EngineError, QueryRejected, RowLimitExceeded
from ml.data.sources import ConnectionFailure, ConnectionSpec, open_source
from ml.data.sql_guard import GuardedQuery, ResultTooLarge, SqlRejected, guard

BACKEND = Path(__file__).resolve().parents[2]
HOSTILE = yaml.safe_load((BACKEND / "tests" / "fixtures" / "hostile_sql.yaml").read_text())
TABLE = "orders"


def cases(dialect: str) -> list[tuple[str, str]]:
    return [(c["sql"].replace("{t}", TABLE), c["code"]) for c in HOSTILE[dialect]]


def allowed(dialect: str) -> list[str]:
    return [q.replace("{t}", TABLE) for q in HOSTILE["allowed"][dialect]]


# -- layer 1, offline ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "sql", "code"),
    [(d, sql, code) for d in ("postgres", "sqlite", "duckdb") for sql, code in cases(d)],
)
def test_every_hostile_query_is_refused_with_its_code(dialect: str, sql: str, code: str) -> None:
    with pytest.raises(SqlRejected) as err:
        guard(sql, dialect)
    assert err.value.code == code, err.value.reason


@pytest.mark.parametrize(
    ("dialect", "sql"), [(d, q) for d in ("postgres", "sqlite", "duckdb") for q in allowed(d)]
)
def test_ordinary_analytical_queries_pass(dialect: str, sql: str) -> None:
    q = guard(sql, dialect)
    assert q.dialect == dialect and q.sql


def test_the_fixture_has_the_matrix_the_issue_asks_for() -> None:
    codes = {code for _, code in cases("postgres")}
    assert len(cases("postgres")) >= 15
    assert {"dml", "ddl", "multi_statement", "copy", "locking_clause", "denied_function"} <= codes
    sqls = " ".join(sql for sql, _ in cases("postgres"))
    for needle in ("WITH d AS (DELETE", "COPY", "TO PROGRAM", "pg_sleep(1000)", "FOR UPDATE"):
        assert needle in sqls
    assert "set_config" in sqls


def test_the_table_allowlist_admits_only_introspected_tables_and_ctes() -> None:
    tables = {"orders", "public.orders"}
    guard("SELECT * FROM orders", "postgres", allowed_tables=tables)
    guard("SELECT * FROM public.orders", "postgres", allowed_tables=tables)
    guard("WITH x AS (SELECT * FROM orders) SELECT * FROM x", "postgres", allowed_tables=tables)
    guard("SELECT * FROM generate_series(1, 3) AS g", "postgres", allowed_tables=tables)
    for sql in (
        "SELECT * FROM customers",
        "SELECT * FROM other.orders",
        "SELECT * FROM orders JOIN pg_catalog.pg_class ON true",
        "SELECT * FROM orders WHERE id IN (SELECT id FROM secrets)",
    ):
        with pytest.raises(SqlRejected) as err:
            guard(sql, "postgres", allowed_tables=tables)
        assert err.value.code == "table_not_allowed"


def test_what_runs_is_rendered_from_the_checked_tree() -> None:
    q = guard("select   id from orders -- a comment\n", "postgres")
    assert q.sql == "SELECT id FROM orders"
    assert q.tables == frozenset({"orders"})
    # Comments are dropped, so none can close early and smuggle a statement in.
    sneaky = guard("SELECT 1 AS a -- */ ; DROP TABLE orders\n", "sqlite")
    assert sneaky.sql == "SELECT 1 AS a"
    assert len(q.sha256) == 64


@pytest.mark.parametrize("sql", ["", "   ", "SELEC 1 FROM", "SELECT " + "1 + " * 40_000 + "1"])
def test_empty_unparseable_or_huge_sql_is_refused(sql: str) -> None:
    with pytest.raises(SqlRejected) as err:
        guard(sql, "postgres")
    assert err.value.code in {"empty", "parse_error", "too_long"}


def test_only_guard_builds_a_guarded_query() -> None:
    """``GuardedQuery(...)`` is constructed in sql_guard.py and in tests only."""
    builders = [
        f"{path.relative_to(BACKEND)}:{n}"
        for folder in ("ml", "app")
        for path in (BACKEND / folder).rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\bGuardedQuery\(", line) and path.name != "sql_guard.py"
    ]
    assert builders == []


# -- SQLite and DuckDB files ---------------------------------------------------------


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)")
    con.executemany(
        f"INSERT INTO {TABLE} VALUES (?, ?, ?)", [(i, f"c{i % 3}", i * 2.5) for i in range(1, 21)]
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def duckdb_file(tmp_path: Path) -> Path:
    path = tmp_path / "shop.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        f"CREATE TABLE {TABLE} AS SELECT i AS id, 'c' || (i % 3) AS customer, "
        "i * 2.5 AS amount FROM range(1, 21) t(i)"
    )
    con.close()
    return path


def _source(dialect: str, sqlite_file: Path, duckdb_file: Path) -> Any:
    path = sqlite_file if dialect == "sqlite" else duckdb_file
    return open_source(ConnectionSpec(dialect, str(path)))  # type: ignore[arg-type]


def _snapshot(src: Any) -> list[tuple[Any, ...]]:
    return sql_guard.catalog(src, f"SELECT id, customer, amount FROM {TABLE} ORDER BY id")


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_hostile_queries_are_refused_by_a_real_file_database(
    dialect: str, sqlite_file: Path, duckdb_file: Path
) -> None:
    src = _source(dialect, sqlite_file, duckdb_file)
    before = _snapshot(src)
    for sql, code in cases(dialect):
        with pytest.raises(SqlRejected) as err:
            src.query(sql, limit=100, timeout_s=5)
        assert err.value.code == code, sql
    for sql in allowed(dialect):
        assert src.query(sql, limit=100, timeout_s=5).num_rows > 0
    assert _snapshot(src) == before
    assert not Path("/tmp/mlpilot_guard_attach.db").exists()
    assert not Path("/tmp/mlpilot_guard_out.csv").exists()


WRITES = [
    f"INSERT INTO {TABLE} VALUES (99, 'x', 1)",
    f"UPDATE {TABLE} SET amount = 0",
    f"DELETE FROM {TABLE}",
    f"DROP TABLE {TABLE}",
    "CREATE TABLE sneaky (a INTEGER)",
    f"ALTER TABLE {TABLE} ADD COLUMN extra INTEGER",
]


def _unguarded(src: Any, sql: str) -> Any:
    """Layer 1 switched off: the SQL goes straight into the read-only session."""
    with sql_guard.read_only_session(src, 5) as session:
        return session.fetch_rows(sql, [], max_rows=None, max_bytes=10**6)


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
@pytest.mark.parametrize("sql", WRITES)
def test_with_the_parser_off_the_session_still_refuses_writes(
    dialect: str, sql: str, sqlite_file: Path, duckdb_file: Path
) -> None:
    src = _source(dialect, sqlite_file, duckdb_file)
    before = _snapshot(src)
    with pytest.raises((EngineError, ConnectionFailure)) as err:
        _unguarded(src, sql)
    assert re.search(r"(?i)not authorized|read-?only|readonly", str(err.value)), err.value
    assert _snapshot(src) == before


def test_sqlite_session_refuses_attach_and_settings_without_the_parser(
    sqlite_file: Path, tmp_path: Path
) -> None:
    src = _source("sqlite", sqlite_file, sqlite_file)
    target = tmp_path / "attached.db"
    for sql in (
        f"ATTACH DATABASE '{target}' AS x",
        "PRAGMA query_only = OFF",
        "PRAGMA writable_schema = 1",
        "SELECT load_extension('x')",
    ):
        with pytest.raises(EngineError, match="(?i)not authorized|prohibited"):
            _unguarded(src, sql)
    assert not target.exists()


def test_duckdb_session_runs_one_statement_and_keeps_files_out_of_reach(
    duckdb_file: Path, tmp_path: Path
) -> None:
    src = _source("duckdb", duckdb_file, duckdb_file)
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    with pytest.raises(QueryRejected, match="one statement"):
        _unguarded(src, f"SELECT 1; DROP TABLE {TABLE}")
    for sql in (
        f"SELECT * FROM read_csv('{secret}')",
        f"COPY {TABLE} TO '{tmp_path / 'out.csv'}'",
        "SET enable_external_access = true",
        f"ATTACH '{tmp_path / 'other.duckdb'}' AS o",
    ):
        with pytest.raises(EngineError):
            _unguarded(src, sql)
    assert not (tmp_path / "out.csv").exists() and not (tmp_path / "other.duckdb").exists()


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_row_and_byte_limits_raise_instead_of_truncating(
    dialect: str, sqlite_file: Path, duckdb_file: Path
) -> None:
    src = _source(dialect, sqlite_file, duckdb_file)
    q = guard(f"SELECT * FROM {TABLE}", dialect)
    assert sql_guard.execute(src, q, limit=20, timeout_s=5).num_rows == 20
    with pytest.raises(RowLimitExceeded):
        sql_guard.execute(src, q, limit=19, timeout_s=5)
    with pytest.raises(ResultTooLarge):
        sql_guard.execute(src, q, limit=100, timeout_s=5, max_bytes=64)


def test_execute_takes_only_checked_queries_of_its_own_dialect(sqlite_file: Path) -> None:
    src = _source("sqlite", sqlite_file, sqlite_file)
    with pytest.raises(TypeError):
        sql_guard.execute(src, f"SELECT * FROM {TABLE}", limit=5, timeout_s=5)  # type: ignore[arg-type]
    with pytest.raises(QueryRejected):
        sql_guard.execute(src, guard("SELECT 1", "duckdb"), limit=5, timeout_s=5)
    assert isinstance(guard("SELECT 1", "sqlite"), GuardedQuery)


def test_the_query_log_has_a_hash_duration_and_rows_but_never_the_sql_or_values(
    sqlite_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = _source("sqlite", sqlite_file, sqlite_file)
    caplog.set_level(logging.DEBUG)
    marker = "needle-value-7731"
    src.query(f"SELECT id FROM {TABLE} WHERE customer <> '{marker}'", limit=100, timeout_s=5)
    with pytest.raises(SqlRejected):
        src.query(f"DELETE FROM {TABLE} WHERE customer = '{marker}'", limit=5, timeout_s=5)
    with pytest.raises(SqlRejected):
        src.query("SELEC broken FROM", limit=5, timeout_s=5)  # sqlglot would quote it
    text = caplog.text
    assert marker not in text and "SELEC" not in text and "DELETE" not in text
    executed = [r.getMessage() for r in caplog.records if r.name == "mlpilot.sql"]
    assert any(
        re.search(r"sql executed sha256=[0-9a-f]{16} dialect=sqlite outcome=ok rows=20 ", m)
        and "duration_ms=" in m
        for m in executed
    ), executed
    assert any("sql rejected" in m and "code=dml" in m for m in executed)


def test_live_sources_run_sql_only_through_the_guard() -> None:
    """No driver call outside sql_guard.py: sources and the SQL import only open connections."""
    driver_call = re.compile(r"\.(execute|executemany|executescript|run|execute_unnamed)\(")
    offenders = [
        f"{path.relative_to(BACKEND)}:{n}: {line.strip()}"
        for folder in ("ml/data/sources", "ml/data/ingestion")
        for path in (BACKEND / folder).rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if driver_call.search(line) and "sql_guard." not in line
    ]
    assert offenders == []
