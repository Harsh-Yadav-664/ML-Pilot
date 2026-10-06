"""Interim read-only SQL hardening (sqlglot validation, read-only transaction, row limit)."""

from __future__ import annotations

import sqlite3

import pytest

from ml.data.ingestion.sql_loader import SqlLoader, UnsafeQueryError, redact


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
    with pytest.raises(UnsafeQueryError):
        SqlLoader().load(url, query)
    assert row_count(path) == 50


def test_select_and_cte_work_with_row_limit(sqlite_url):
    url, _ = sqlite_url
    df = SqlLoader().load(
        url, "WITH x AS (SELECT * FROM t WHERE id < 30) SELECT * FROM x", row_limit=10
    )
    assert len(df) == 10
    assert list(df.columns) == ["id", "name"]


def test_redact_removes_password_and_connection_string():
    url = "postgresql://alice:s3cretpw@db.internal:5432/sales"
    msg = f"could not connect to {url} (password s3cretpw rejected)"
    out = redact(msg, url)
    assert "s3cretpw" not in out and url not in out
