"""The old connection-string import goes through the same source and SQL guard (#44)."""

from __future__ import annotations

import sqlite3

import pytest

from ml.data.engine import EngineError, RowLimitExceeded
from ml.data.ingestion.sql_loader import SqlLoader, redact, spec_from_url
from ml.data.sql_guard import SqlRejected


@pytest.fixture
def sqlite_url(tmp_path):
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"n{i}") for i in range(50)])
    return f"sqlite:///{path}", path


def row_count(path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO t VALUES (99, 'x')",
        "UPDATE t SET name = 'x'",
        "DROP TABLE t",
        "DELETE FROM t",
        "WITH x AS (DELETE FROM t RETURNING *) SELECT 1",
        "SELECT 1; DROP TABLE t",
        "CREATE TABLE u AS SELECT * FROM t",
        "SELECT * INTO u FROM t",
    ],
)
def test_unsafe_queries_are_refused_and_data_is_untouched(sqlite_url, query):
    url, path = sqlite_url
    with pytest.raises(SqlRejected):
        SqlLoader().load(url, query)
    assert row_count(path) == 50


def test_select_and_cte_work_and_the_row_limit_raises_instead_of_truncating(sqlite_url):
    url, _ = sqlite_url
    query = "WITH x AS (SELECT * FROM t WHERE id < 30) SELECT * FROM x"
    df = SqlLoader().load(url, query, row_limit=30)
    assert len(df) == 30
    assert list(df.columns) == ["id", "name"]
    with pytest.raises(RowLimitExceeded):
        SqlLoader().load(url, query, row_limit=10)


def test_connection_strings_become_connection_specs():
    pg = spec_from_url(
        "postgresql+psycopg2://alice:s3cretpw@db.internal:6543/sales?sslmode=require"
    )
    assert (pg.dialect, pg.host, pg.port, pg.username, pg.database, pg.ssl_mode) == (
        "postgres",
        "db.internal",
        6543,
        "alice",
        "sales",
        "require",
    )
    assert pg.password == "s3cretpw" and "s3cretpw" not in repr(pg)
    assert spec_from_url("sqlite:////data/shop.db").database == "/data/shop.db"
    assert spec_from_url("duckdb:///shop.duckdb").dialect == "duckdb"
    for bad in ("mysql://u:p@h/db", "sqlite://", "sqlite:///:memory:", "not a url"):
        with pytest.raises(EngineError):
            spec_from_url(bad)


def test_redact_removes_password_and_connection_string():
    url = "postgresql://alice:s3cretpw@db.internal:5432/sales"
    msg = f"could not connect to {url} (password s3cretpw rejected)"
    out = redact(msg, url)
    assert "s3cretpw" not in out and url not in out
