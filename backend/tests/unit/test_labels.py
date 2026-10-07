"""Labels at cutoff dates (#50): a hand-checked shop, the demo database against pandas, dropped cutoffs.

The tiny shop (tests/fixtures/tiny_shop.sql) has five customers and three cutoffs; every expected
row below was worked out by hand from the comments in that file.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sqlglot import exp

from ml.data.engine import arrow_to_pandas
from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.labels import (
    LabelError,
    compile_labels,
    counts_select,
    label_windows,
    required_columns,
    window_end_of,
)
from ml.tasks.spec import from_yaml
from tests.fixtures.demo_db import demo_duckdb

TINY_SQL = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_shop.sql"
DATA_ENDS = datetime(2024, 7, 1, tzinfo=UTC)

NO_ORDER = """
name: no_order_30d
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - exists: {table: orders}
target:
  type: binary
  expression: {table: orders, agg: count, where: "status != 'cancelled'", compare: "= 0"}
horizon: 30d
cutoffs: {start: 2024-03-01, end: 2024-05-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-05-01}
metric: roc_auc
"""


def open_duckdb(path: Path) -> Any:
    return open_source(ConnectionSpec(dialect="duckdb", database=str(path)))


@pytest.fixture
def tiny(tmp_path: Path) -> tuple[Any, SchemaGraph]:
    path = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(path))
    con.execute(TINY_SQL.read_text())
    con.close()
    source = open_duckdb(path)
    return source, build_schema_graph(source)


def resolver(graph: SchemaGraph):
    keys = {t.key for t in graph.tables}

    def resolve(key: str) -> exp.Table:
        assert key in keys
        return exp.Table(this=exp.to_identifier(key, quoted=True))

    return resolve


def run(source: Any, graph: SchemaGraph, yaml_text: str, as_of: datetime = DATA_ENDS):
    spec = from_yaml(yaml_text)
    compiled = compile_labels(spec, graph, "duckdb", resolver(graph), as_of)
    frame = arrow_to_pandas(source.query(compiled.sql, limit=1_000_000, timeout_s=60))
    return spec, compiled, frame


def rows(frame: pd.DataFrame) -> list[tuple[int, str, int]]:
    return sorted(
        (int(r.entity_id), pd.Timestamp(r.cutoff_time).strftime("%Y-%m-%d"), int(r.label))
        for r in frame.itertuples()
    )


def test_hand_checked_labels(tiny: tuple[Any, SchemaGraph]) -> None:
    source, graph = tiny
    _, compiled, frame = run(source, graph, NO_ORDER)
    # label 1 = no (non-cancelled) order in (cutoff, cutoff + 30d]
    assert rows(frame) == sorted(
        [
            # 03-01: c3 had no order before it, c4 did not exist yet
            (1, "2024-03-01", 0),  # order exactly at the window end (03-31 00:00) is inside
            (2, "2024-03-01", 1),
            (5, "2024-03-01", 1),  # the cancelled 03-15 order does not count
            # 04-01: c2's order exactly at the cutoff is not in the window, so c2 has none
            (1, "2024-04-01", 1),
            (2, "2024-04-01", 1),
            (4, "2024-04-01", 0),
            (5, "2024-04-01", 1),
            # 05-01: the 04-01 00:00 order is history now; c2 orders again on 05-15
            (1, "2024-05-01", 1),
            (2, "2024-05-01", 0),
            (4, "2024-05-01", 1),
            (5, "2024-05-01", 1),
        ]
    )
    assert list(frame.columns) == ["entity_id", "cutoff_time", "label", "label_window_end"]
    assert len(compiled.kept) == 3 and compiled.dropped == []


def test_the_future_never_makes_an_entity_eligible(tiny: tuple[Any, SchemaGraph]) -> None:
    source, graph = tiny
    _, _, frame = run(source, graph, NO_ORDER)
    assert 3 not in set(frame.entity_id)  # its only order is on 06-20, after every cutoff


def test_not_exists_rule_and_regression_sum(tiny: tuple[Any, SchemaGraph]) -> None:
    source, graph = tiny
    text = """
name: spend
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - not_exists: {table: orders, where: "status = 'cancelled'"}
target:
  type: regression
  expression: {table: orders, agg: count, where: "status != 'cancelled'"}
horizon: 30d
cutoffs: {start: 2024-03-01, end: 2024-05-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-05-01}
"""
    _, _, frame = run(source, graph, text)
    got = sorted(
        (int(r.entity_id), pd.Timestamp(r.cutoff_time).strftime("%m-%d"), float(r.label))
        for r in frame.itertuples()
    )
    # c5's cancelled order (03-15) is after the first cutoff, so c5 is still eligible there,
    # and from 04-01 on the not_exists test sees it and drops c5.
    assert got == [
        (1, "03-01", 1.0),
        (1, "04-01", 0.0),
        (1, "05-01", 0.0),
        (2, "03-01", 0.0),
        (2, "04-01", 0.0),
        (2, "05-01", 1.0),
        (3, "03-01", 0.0),
        (3, "04-01", 0.0),
        (3, "05-01", 0.0),
        (4, "04-01", 1.0),
        (4, "05-01", 0.0),
        (5, "03-01", 0.0),
    ]


def test_incomplete_windows_are_dropped_and_reported(
    tiny: tuple[Any, SchemaGraph],
) -> None:
    source, graph = tiny
    # The data ends 2024-05-20: the 05-01 window (to 05-31) is not complete, 04-01's (to 05-01) is.
    as_of = datetime(2024, 5, 20, tzinfo=UTC)
    _, compiled, frame = run(source, graph, NO_ORDER, as_of)
    assert [w.cutoff.date().isoformat() for w in compiled.kept] == ["2024-03-01", "2024-04-01"]
    assert [w.cutoff.date().isoformat() for w in compiled.dropped] == ["2024-05-01"]
    assert {r[1] for r in rows(frame)} == {"2024-03-01", "2024-04-01"}


def test_a_window_ending_exactly_at_as_of_is_complete() -> None:
    spec = from_yaml(NO_ORDER)
    kept, dropped = label_windows(spec, datetime(2024, 5, 31, tzinfo=UTC))
    assert len(kept) == 3 and dropped == []
    kept, dropped = label_windows(spec, datetime(2024, 5, 30, 23, 59, tzinfo=UTC))
    assert len(kept) == 2 and len(dropped) == 1


def test_no_complete_cutoff_is_an_error(tiny: tuple[Any, SchemaGraph]) -> None:
    _, graph = tiny
    with pytest.raises(LabelError, match="no labels are complete"):
        compile_labels(
            from_yaml(NO_ORDER),
            graph,
            "duckdb",
            resolver(graph),
            datetime(2024, 3, 15, tzinfo=UTC),
        )


def test_expression_sql_waits_for_the_point_in_time_guard(
    tiny: tuple[Any, SchemaGraph],
) -> None:
    _, graph = tiny
    text = NO_ORDER.replace(
        'expression: {table: orders, agg: count, where: "status != \'cancelled\'", compare: "= 0"}',
        'expression_sql: "SELECT 1"',
    )
    with pytest.raises(LabelError, match="expression_sql|#51|spec has errors"):
        compile_labels(from_yaml(text), graph, "duckdb", resolver(graph), DATA_ENDS)


def test_window_end_can_be_earlier_than_the_horizon() -> None:
    text = NO_ORDER.replace(
        "  type: binary\n",
        '  type: binary\n  window: {start: ":cutoff", end: ":cutoff + interval \'7 days\'"}\n',
    )
    spec = from_yaml(text)
    assert window_end_of(spec, date(2024, 3, 1)) == datetime(2024, 3, 8, tzinfo=UTC)


def test_required_columns_name_what_a_snapshot_must_hold(
    tiny: tuple[Any, SchemaGraph],
) -> None:
    _, graph = tiny
    need = required_columns(from_yaml(NO_ORDER), graph)
    assert need == {
        "customers": ["customer_id", "signup_at"],
        "orders": ["customer_id", "ordered_at", "status"],
    }


def test_counts_per_cutoff(tiny: tuple[Any, SchemaGraph]) -> None:
    source, graph = tiny
    _, compiled, _ = run(source, graph, NO_ORDER)
    sql = counts_select(compiled.select, binary=True).sql(dialect="duckdb")
    frame = arrow_to_pandas(source.query(sql, limit=100, timeout_s=60))
    assert [(int(e), int(p)) for e, p in zip(frame.eligible, frame.positives, strict=True)] == [
        (3, 2),
        (4, 3),
        (4, 3),
    ]


# -- the demo database against an independent pandas computation -------------------------------

CHURN = """
name: churn_30d
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
  - exists: {table: orders, where: "ordered_at >= :cutoff - interval '90 days'"}
target:
  type: binary
  expression: {table: orders, agg: count, where: "status != 'cancelled'", compare: "= 0"}
horizon: 30d
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
metric: pr_auc
"""


def test_churn_labels_match_pandas_row_for_row(tmp_path: Path) -> None:
    path = demo_duckdb(tmp_path / "demo.duckdb")
    source = open_duckdb(path)
    graph = build_schema_graph(source)
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    _, compiled, got = run(source, graph, CHURN, as_of)

    customers = arrow_to_pandas(source.query("SELECT * FROM customers", limit=10**6, timeout_s=60))
    orders = arrow_to_pandas(source.query("SELECT * FROM orders", limit=10**7, timeout_s=60))
    customers["signup_at"] = pd.to_datetime(customers["signup_at"])
    orders["ordered_at"] = pd.to_datetime(orders["ordered_at"])
    expected_rows = []
    for w in compiled.kept:
        cutoff = pd.Timestamp(w.cutoff.replace(tzinfo=None))
        end = pd.Timestamp(w.window_end.replace(tzinfo=None))
        lower = cutoff - pd.Timedelta(days=90)
        for cid, signup in zip(customers.customer_id, customers.signup_at, strict=True):
            if not signup < cutoff:
                continue
            mine = orders[orders.customer_id == cid]
            if not ((mine.ordered_at >= lower) & (mine.ordered_at < cutoff)).any():
                continue  # exists rule: an order in the 90 days before the cutoff
            future = mine[
                (mine.ordered_at > cutoff) & (mine.ordered_at <= end) & (mine.status != "cancelled")
            ]
            expected_rows.append((int(cid), cutoff.strftime("%Y-%m-%d"), int(len(future) == 0)))
    assert len(expected_rows) > 500  # the comparison is not vacuous
    assert rows(got) == sorted(expected_rows)
    assert {r[2] for r in expected_rows} == {0, 1}


def test_a_sqlite_snapshot_gives_the_same_counts_as_typed_timestamps(tmp_path: Path) -> None:
    """SQLite keeps times as text and the snapshot keeps them as delivered; the labels agree."""
    from ml.data.engine import DuckDBSource
    from ml.data.snapshot import create_snapshot
    from ml.data.workspace import Workspace
    from ml.tasks.labels import run_on_snapshot, run_on_source
    from tests.fixtures.demo_db import demo_sqlite

    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    spec = from_yaml(CHURN)
    typed = open_duckdb(demo_duckdb(tmp_path / "demo.duckdb"))
    typed_run = run_on_source(spec, build_schema_graph(typed), typed, as_of)

    sqlite = open_source(
        ConnectionSpec(dialect="sqlite", database=str(demo_sqlite(tmp_path / "demo.sqlite")))
    )
    graph = build_schema_graph(sqlite)
    workspace = Workspace(DuckDBSource(tmp_path / "work.duckdb"))
    try:
        info = create_snapshot(
            sqlite, graph, workspace, tables=["customers", "orders"], as_of=as_of
        )
        snap_run = run_on_snapshot(spec, graph, workspace, info.id, info.description, task_id="t-1")
        assert snap_run.counts == typed_run.counts
        assert snap_run.table is not None
        ((rows_in_table,),) = workspace.source.execute(
            f"SELECT count(*) FROM {snap_run.table.sql()}"
        )
        assert rows_in_table == snap_run.total_rows == typed_run.total_rows > 1000
        print(f"\nsqlite snapshot == typed duckdb: {rows_in_table} label rows")
    finally:
        workspace.source.close()
