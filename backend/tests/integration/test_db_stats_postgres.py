"""Column statistics on a real Postgres (#96): the demo database against pandas, and 5M rows."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from ml.data.engine import TableRef
from ml.data.profiling.db_stats import StatsConfig, profile_table
from ml.data.sources import open_source
from tests.fixtures.demo_db import demo_sqlite
from tests.fixtures.postgres import pg_spec
from tests.integration.conftest import SCHEMA, admin_connection
from tests.unit.test_db_stats import (
    BIG_ROWS,
    BIG_TIME_LIMIT_S,
    TABLES,
    assert_big_profile,
    check_column,
)


def test_every_column_of_every_demo_table_matches_pandas_in_postgres(
    demo_pg, tmp_path: Path
) -> None:
    _, ro, _ = demo_pg
    source = open_source(ro)
    with sqlite3.connect(demo_sqlite(tmp_path / "ref.sqlite")) as con:
        frames = {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in TABLES}
    compared = 0
    for name in TABLES:
        stats = profile_table(source, TableRef(name, SCHEMA))
        assert stats.row_count == stats.profiled_rows == len(frames[name]) and not stats.sampled
        for col in stats.columns:
            check_column(col, frames[name][col.name], exact=True, tol=1e-6)
            compared += 1
    print(f"\npostgres: {compared} columns of {len(TABLES)} tables match pandas within 1e-6")
    assert stats.table == f"{SCHEMA}.support_tickets"


@pytest.fixture
def big_table() -> Iterator[str]:
    spec = pg_spec()
    admin = admin_connection(spec)
    admin.run("DROP TABLE IF EXISTS mlpilot_big_stats")
    admin.run(
        "CREATE TABLE mlpilot_big_stats AS SELECT i AS id, i % 1000 AS value, "
        f"CASE WHEN i % 3 = 1 THEN 'a' ELSE 'b' END AS flag FROM generate_series(1, {BIG_ROWS}) i"
    )
    yield "mlpilot_big_stats"
    admin.run("DROP TABLE IF EXISTS mlpilot_big_stats")
    admin.close()


def test_a_five_million_row_postgres_table_is_profiled_by_sampling_within_the_limit(
    big_table: str,
) -> None:
    source = open_source(pg_spec())
    config = StatsConfig(timeout_s=BIG_TIME_LIMIT_S, budget_s=BIG_TIME_LIMIT_S)
    started = time.monotonic()
    stats = profile_table(source, TableRef(big_table, "public"), config)
    assert_big_profile(stats, time.monotonic() - started, "postgres")
    assert stats.sample_method == "system"
