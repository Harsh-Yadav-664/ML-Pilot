"""Column statistics of database tables (#96): exact against pandas, sampled within 5%, aggregates only.

The demo database is profiled as a SQLite and a DuckDB file and every column of every table is
compared with a pandas computation on the same data. Large tables are generated in the file
itself so the sampling and the time limit are exercised on millions of rows.
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from ml.data import sql_guard
from ml.data.engine import QueryTimeout, TableRef
from ml.data.profiling.db_stats import (
    ColumnStats,
    StatsConfig,
    TableStats,
    profile_table,
    semantic_type,
)
from ml.data.sources import ConnectionSpec, SourceWithChecks, open_source
from tests.fixtures.demo_db import demo_duckdb, demo_sqlite

TABLES = [
    "customer_status_snapshot",
    "customers",
    "marketing_emails",
    "order_items",
    "orders",
    "products",
    "refunds",
    "sessions",
    "support_tickets",
]


def source_for(path: Path, dialect: str) -> SourceWithChecks:
    return open_source(ConnectionSpec(dialect=dialect, database=str(path)))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("demo")
    return {
        "sqlite": demo_sqlite(folder / "demo.sqlite"),
        "duckdb": demo_duckdb(folder / "demo.duckdb"),
    }


@pytest.fixture(scope="module")
def reference(demo_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """The demo tables as pandas frames (read from the SQLite copy)."""
    with sqlite3.connect(demo_files["sqlite"]) as con:
        return {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in TABLES}


def close(a: float | None, b: float, tol: float) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def check_column(col: ColumnStats, series: pd.Series, *, exact: bool, tol: float) -> None:
    where = f"{col.name} ({col.semantic_type})"
    if exact:
        assert col.non_null == series.notna().sum(), where
        assert col.distinct == series.nunique(), where
        assert close(col.null_fraction, series.isna().mean(), 1e-9), where
    else:
        assert close(col.null_fraction, series.isna().mean(), 0.02), where
    kind = pd.api.types.is_numeric_dtype(series)
    if col.semantic_type in ("numeric", "id") and kind:
        values = series.astype(float)
        if exact:  # the extremes of a sample are not the extremes of the table
            assert close(col.min, values.min(), tol), where
            assert close(col.max, values.max(), tol), where
        assert close(col.mean, values.mean(), tol), where
        assert close(col.stddev, values.std(ddof=1), tol), where
        for label, q in (("p1", 0.01), ("p50", 0.5), ("p99", 0.99)):
            assert close(getattr(col, label), values.quantile(q), tol), (where, label)
    elif col.semantic_type == "boolean":
        assert close(col.mean, series.astype(float).mean(), tol), where
    elif col.semantic_type == "time" and exact:
        assert col.time_min == str(series.min()) and col.time_max == str(series.max()), where


# -- exact statistics on the demo database -----------------------------------------------


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_every_column_of_every_demo_table_matches_pandas(
    demo_files: dict[str, Path], reference: dict[str, pd.DataFrame], dialect: str
) -> None:
    source = source_for(demo_files[dialect], dialect)
    compared = 0
    for name in TABLES:
        stats = profile_table(source, TableRef(name, "main"))
        frame = reference[name]
        assert stats.row_count == stats.profiled_rows == len(frame)
        assert not stats.sampled and stats.sample_fraction is None
        assert [c.name for c in stats.columns] == list(frame.columns)
        for col in stats.columns:
            check_column(col, frame[col.name], exact=True, tol=1e-6)
            compared += 1
    print(f"\n{dialect}: {compared} columns of {len(TABLES)} tables match pandas within 1e-6")


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_top_values_are_counts_of_the_most_frequent_categories(
    demo_files: dict[str, Path], reference: dict[str, pd.DataFrame], dialect: str
) -> None:
    source = source_for(demo_files[dialect], dialect)
    stats = profile_table(source, TableRef("customers", "main"), StatsConfig(top_k=3))
    country = next(c for c in stats.columns if c.name == "country")
    expected = reference["customers"]["country"].value_counts()
    top = [(v.value, v.count) for v in country.top_values]
    assert len(top) == 3
    assert [c for _, c in top] == sorted(expected.tolist(), reverse=True)[:3]
    assert all(expected[v] == c for v, c in top)
    for other in stats.columns:
        if other.semantic_type != "categorical":
            assert other.top_values == [], other.name


def test_semantic_types_of_the_demo_columns(demo_files: dict[str, Path]) -> None:
    source = source_for(demo_files["sqlite"], "sqlite")
    types = {
        (t, c.name): c.semantic_type
        for t in ("customers", "orders", "support_tickets")
        for c in profile_table(source, TableRef(t, "main")).columns
    }
    assert types[("customers", "customer_id")] == "id"
    assert types[("orders", "customer_id")] == "id"  # a foreign key
    assert types[("customers", "email")] == "text"  # one value per customer
    assert types[("customers", "country")] == "categorical"
    assert types[("customers", "is_churned")] == "boolean"
    assert types[("customers", "signup_at")] == "time"
    assert types[("orders", "total")] == "numeric"
    assert types[("support_tickets", "satisfaction")] == "numeric"


@pytest.mark.parametrize(
    ("name", "kind", "key", "nn", "distinct", "lo", "hi", "expected"),
    [
        ("id", "integer", False, 100, 100, 1, 100, "id"),
        ("customerid", "integer", False, 100, 10, 1, 9, "numeric"),  # lower-case form: not a hint
        ("flag", "integer", False, 100, 2, 0, 1, "boolean"),
        ("n", "integer", False, 100, 90, 0, 500, "numeric"),
        ("price", "float", False, 100, 90, 1.0, 9.5, "numeric"),
        ("created", "timestamp", False, 100, 100, None, None, "time"),
        ("ok", "boolean", False, 100, 2, None, None, "boolean"),
        ("plan", "text", False, 1000, 3, None, None, "categorical"),
        ("email", "text", False, 1000, 1000, None, None, "text"),
        ("note", "text", False, 1000, 400, None, None, "text"),
        ("ref_uuid", "text", False, 1000, 1000, None, None, "id"),
        ("code", "text", True, 1000, 1000, None, None, "id"),
        ("blob", "other", False, 10, 10, None, None, "other"),
    ],
)
def test_semantic_type(name, kind, key, nn, distinct, lo, hi, expected) -> None:
    assert (
        semantic_type(
            name, kind, is_key=key, non_null=nn, distinct=distinct, minimum=lo, maximum=hi
        )
        == expected
    )


def test_wide_tables_are_profiled_in_chunks_with_the_same_result(
    demo_files: dict[str, Path],
) -> None:
    source = source_for(demo_files["duckdb"], "duckdb")
    ref = TableRef("orders", "main")
    whole = profile_table(source, ref)
    chunked = profile_table(source, ref, StatsConfig(columns_per_query=2))
    assert chunked.columns == whole.columns


# -- sampling ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "method"), [("sqlite", "rowid_modulo"), ("duckdb", "bernoulli")]
)
def test_a_sampled_profile_says_so_and_stays_within_five_percent(
    demo_files: dict[str, Path],
    reference: dict[str, pd.DataFrame],
    dialect: str,
    method: str,
) -> None:
    source = source_for(demo_files[dialect], dialect)
    config = StatsConfig(sample_above_rows=1000, sample_rows=4000)
    stats = profile_table(source, TableRef("orders", "main"), config)
    assert stats.sampled and stats.sample_method == method
    assert stats.row_count == 12032 and 0.2 < stats.sample_fraction <= 0.4  # type: ignore[operator]
    assert 0.8 * stats.sample_fraction * 12032 < stats.profiled_rows  # type: ignore[operator]
    assert stats.profiled_rows < 1.2 * stats.sample_fraction * 12032  # type: ignore[operator]
    frame = reference["orders"]
    for col in stats.columns:
        if col.semantic_type == "numeric":
            check_column(col, frame[col.name], exact=False, tol=0.05)
    assert close(
        next(c for c in stats.columns if c.name == "total").mean, frame["total"].mean(), 0.05
    )


def test_a_sqlite_table_without_rowid_falls_back_to_the_first_rows(tmp_path: Path) -> None:
    path = tmp_path / "norowid.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v REAL) WITHOUT ROWID")
        con.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 0.5) for i in range(1, 5001)])
    stats = profile_table(
        source_for(path, "sqlite"),
        TableRef("t", "main"),
        StatsConfig(sample_above_rows=1000, sample_rows=1000),
    )
    assert stats.sampled and stats.sample_method == "head" and stats.profiled_rows == 1000


# -- nothing but aggregates leaves the database ------------------------------------------


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_no_stats_query_returns_more_than_one_row_per_group(
    demo_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, dialect: str
) -> None:
    calls: list[tuple[str, int, int]] = []
    real = sql_guard.execute

    def spy(source: Any, q: Any, *, limit: int, **kwargs: Any):
        table = real(source, q, limit=limit, **kwargs)
        calls.append((q.sql, limit, table.num_rows))
        return table

    monkeypatch.setattr(sql_guard, "execute", spy)
    config = StatsConfig(top_k=5)
    source = source_for(demo_files[dialect], dialect)
    for name in TABLES:
        profile_table(source, TableRef(name, "main"), config)
    aggregates = [c for c in calls if "GROUP BY" not in c[0]]
    top_values = [c for c in calls if "GROUP BY" in c[0]]
    assert aggregates and top_values
    assert all(limit == 1 and rows == 1 for _, limit, rows in aggregates)
    assert all(limit == 5 and rows <= 5 for _, limit, rows in top_values)
    assert all("COUNT(" in sql for sql, _, _ in calls if "OFFSET" not in sql)
    print(
        f"\n{dialect}: {len(calls)} stats queries; "
        f"{len(aggregates)} aggregate queries returned 1 row each, "
        f"{len(top_values)} top-value queries returned at most 5 rows"
    )


def test_a_stats_query_is_a_guarded_select(demo_files: dict[str, Path]) -> None:
    """Profiling a table the database does not have is refused by the guard's table list."""
    source = source_for(demo_files["sqlite"], "sqlite")
    with pytest.raises(Exception, match="not_a_table|does not exist|no such table"):
        profile_table(source, TableRef("not_a_table", "main"))


def test_running_out_of_time_raises_instead_of_returning_partial_statistics(
    demo_files: dict[str, Path],
) -> None:
    source = source_for(demo_files["duckdb"], "duckdb")
    with pytest.raises(QueryTimeout, match="budget"):
        profile_table(source, TableRef("orders", "main"), StatsConfig(budget_s=0))


# -- big tables ----------------------------------------------------------------------------

BIG_ROWS = 5_000_000
BIG_TIME_LIMIT_S = 60.0


def assert_big_profile(stats: TableStats, seconds: float, label: str) -> None:
    print(
        f"\n{label}: profiled {stats.row_count:,} rows via {stats.sample_method} sample of "
        f"{stats.profiled_rows:,} in {seconds:.1f} s (limit {BIG_TIME_LIMIT_S:g} s)"
    )
    assert close(stats.row_count, BIG_ROWS, 0.02) and stats.sampled  # Postgres may estimate
    assert 0.9e6 < stats.profiled_rows < 1.1e6
    assert seconds < BIG_TIME_LIMIT_S
    by_name = {c.name: c for c in stats.columns}
    # value = id % 1000: mean 499.5, stddev about 288.7; the id column runs 1..5M.
    assert close(by_name["value"].mean, 499.5, 0.05)
    assert close(by_name["value"].stddev, 288.68, 0.05)
    assert close(by_name["value"].p50, 499.5, 0.05)
    assert close(by_name["id"].mean, (BIG_ROWS + 1) / 2, 0.05)
    assert by_name["id"].semantic_type == "id" and by_name["value"].semantic_type == "numeric"
    assert by_name["flag"].semantic_type == "categorical"


def test_a_five_million_row_duckdb_table_is_profiled_by_sampling_within_the_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "big.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE big AS SELECT i + 1 AS id, (i + 1) % 1000 AS value, "
        f"CASE WHEN i % 3 = 0 THEN 'a' ELSE 'b' END AS flag FROM range({BIG_ROWS}) t(i)"
    )
    con.close()
    config = StatsConfig(timeout_s=BIG_TIME_LIMIT_S, budget_s=BIG_TIME_LIMIT_S)
    started = time.monotonic()
    stats = profile_table(source_for(path, "duckdb"), TableRef("big", "main"), config)
    assert_big_profile(stats, time.monotonic() - started, "duckdb")


def test_a_five_million_row_sqlite_table_is_profiled_by_sampling_within_the_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "big.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, value INTEGER, flag TEXT)")
        con.execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) "
            "INSERT INTO big SELECT i, i % 1000, CASE WHEN i % 3 = 1 THEN 'a' ELSE 'b' END FROM n",
            (BIG_ROWS,),
        )
    config = StatsConfig(timeout_s=BIG_TIME_LIMIT_S, budget_s=BIG_TIME_LIMIT_S)
    started = time.monotonic()
    stats = profile_table(source_for(path, "sqlite"), TableRef("big", "main"), config)
    assert_big_profile(stats, time.monotonic() - started, "sqlite")
