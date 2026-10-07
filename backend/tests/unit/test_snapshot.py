"""Data versions for live databases (#97): same data, same id; nothing after as_of; live mode.

The demo database (#47) is read as a SQLite and a DuckDB file. Each test works on its own copy,
so changing the "company's" data (a new order, rows dated in the future) cannot affect another test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from ml.data import snapshot as snap
from ml.data import sql_guard
from ml.data.engine import DuckDBSource, EngineError
from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.snapshot import (
    SnapshotError,
    SnapshotLimits,
    create_snapshot,
    live_version,
    plan_tables,
    select_sql,
)
from ml.data.sources import ConnectionSpec, open_source
from ml.data.sources.base import SourceWithChecks
from ml.data.workspace import Workspace
from tests.fixtures.demo_db import demo_duckdb, demo_sqlite

AS_OF = datetime(2024, 6, 30, 12, 0, 0, tzinfo=UTC)
FUTURE = "2999-01-01 00:00:00"
TABLES = ["customers", "orders", "sessions"]


class Demo:
    """One copy of the demo database and what is needed to read and change it."""

    def __init__(self, dialect: str, path: Path) -> None:
        self.dialect, self.path = dialect, path

    def source(self) -> SourceWithChecks:
        return open_source(ConnectionSpec(dialect=self.dialect, database=str(self.path)))  # type: ignore[arg-type]

    def graph(self) -> SchemaGraph:
        return build_schema_graph(self.source())

    def run(self, sql: str, params: list[Any] | None = None) -> None:
        """Change the database directly (the company's application writing to it)."""
        if self.dialect == "sqlite":
            with sqlite3.connect(self.path) as con:
                con.execute(sql, params or [])
        else:
            con = duckdb.connect(str(self.path))
            try:
                con.execute(sql, params or [])
            finally:
                con.close()

    def add_order(self, order_id: int, when: str) -> None:
        self.run("INSERT INTO orders VALUES (?, 1, ?, 'completed', 9.99)", [order_id, when])


@pytest.fixture(params=["sqlite", "duckdb"])
def demo(request: pytest.FixtureRequest, tmp_path: Path) -> Demo:
    build: Callable[..., Path] = demo_sqlite if request.param == "sqlite" else demo_duckdb
    return Demo(request.param, build(tmp_path / f"demo.{request.param}"))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(DuckDBSource(tmp_path / "work.duckdb"))
    yield ws  # type: ignore[misc]
    ws.source.close()


def take(demo: Demo, ws: Workspace, **kw: Any) -> snap.DataVersionInfo:
    return create_snapshot(
        demo.source(), demo.graph(), ws, tables=kw.pop("tables", TABLES), as_of=AS_OF, **kw
    )


def in_workspace(ws: Workspace, info: snap.DataVersionInfo, key: str) -> list[tuple[Any, ...]]:
    ref = ws.snapshot_tables(info.id)[key]
    return ws.source.execute(f"SELECT * FROM {ref.sql()}")


# -- acceptance (a): the id is a function of the data --------------------------------------


def test_two_snapshots_of_an_unchanged_database_have_the_same_id(
    demo: Demo, workspace: Workspace
) -> None:
    first = take(demo, workspace)
    second = take(demo, workspace)
    assert first.id == second.id and len(first.id) == 64
    assert first.created and not second.created
    assert first.description.tables["orders"].rows > 1000
    print(f"\n{demo.dialect}: two snapshots -> {first.short_hash} == {second.short_hash}")


def test_a_new_order_before_as_of_changes_the_id(demo: Demo, workspace: Workspace) -> None:
    before = take(demo, workspace)
    demo.add_order(900_001, "2024-06-01 10:00:00")
    after = take(demo, workspace)
    assert after.id != before.id and after.created
    assert after.description.tables["orders"].rows == before.description.tables["orders"].rows + 1
    # Only the orders table changed; the other tables are as they were.
    for key in ("customers", "sessions"):
        assert after.description.tables[key].checksum == before.description.tables[key].checksum
    print(f"\n{demo.dialect}: one inserted order -> {before.short_hash} != {after.short_hash}")


def test_an_edited_value_changes_the_id_even_when_the_row_count_does_not(
    demo: Demo, workspace: Workspace
) -> None:
    before = take(demo, workspace)
    demo.run("UPDATE orders SET total = total + 1 WHERE order_id = 1")
    after = take(demo, workspace)
    assert after.description.tables["orders"].rows == before.description.tables["orders"].rows
    assert after.id != before.id


def test_the_id_does_not_depend_on_as_of_when_the_data_is_the_same(
    demo: Demo, workspace: Workspace
) -> None:
    a = take(demo, workspace)
    later = create_snapshot(
        demo.source(),
        demo.graph(),
        workspace,
        tables=TABLES,
        as_of=datetime(2024, 6, 30, 13, tzinfo=UTC),
    )
    assert later.id == a.id  # nothing happened in that hour


def test_a_snapshot_is_stored_and_matches_the_source_row_for_row(
    demo: Demo, workspace: Workspace
) -> None:
    info = take(demo, workspace)
    rows = in_workspace(workspace, info, "customers")
    assert len(rows) == info.description.tables["customers"].rows
    source = demo.source()
    q = sql_guard.guard("SELECT count(*) FROM customers", source.dialect)
    total = sql_guard.execute(source, q, limit=1, timeout_s=30).column(0)[0].as_py()
    assert len(rows) <= total  # signups after as_of are not in the copy


# -- acceptance (b): nothing after as_of ---------------------------------------------------


def test_rows_dated_after_as_of_are_in_the_source_but_not_in_the_snapshot(
    demo: Demo, workspace: Workspace
) -> None:
    clean = take(demo, workspace)
    for i, when in enumerate([FUTURE, "2024-06-30 12:00:01", "2030-03-03 03:03:03"]):
        demo.add_order(990_000 + i, when)
    # The canaries are really in the source.
    source = demo.source()
    q = sql_guard.guard("SELECT count(*) FROM orders WHERE order_id >= 990000", source.dialect)
    assert sql_guard.execute(source, q, limit=1, timeout_s=30).column(0)[0].as_py() == 3

    info = take(demo, workspace)
    ids = [r[0] for r in in_workspace(workspace, info, "orders")]
    assert not [i for i in ids if i >= 990_000]
    assert info.id == clean.id  # they are not part of the data version either
    assert str(info.description.tables["orders"].max_event_time) <= "2024-06-30 12:00:00"
    print(f"\n{demo.dialect}: 3 future canary orders in the source, 0 in the snapshot")


def test_a_row_exactly_at_as_of_is_included(demo: Demo, workspace: Workspace) -> None:
    demo.add_order(990_100, "2024-06-30 12:00:00")
    info = take(demo, workspace)
    assert 990_100 in [r[0] for r in in_workspace(workspace, info, "orders")]


def test_the_query_that_reads_the_source_carries_the_cutoff(demo: Demo) -> None:
    plans = plan_tables(demo.graph(), demo.dialect, ["orders"], {"orders": ["order_id"]})
    sql = select_sql(demo.dialect, plans[0], AS_OF)
    assert "2024-06-30 12:00:00" in sql and "<=" in sql
    assert plans[0].columns == ["order_id", "ordered_at"]  # the time column is always read


def test_the_copy_is_checked_after_loading_not_only_in_the_query(
    demo: Demo, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a source ignored the WHERE clause, the snapshot would be refused, not kept."""
    demo.add_order(990_200, FUTURE)
    monkeypatch.setattr(
        snap, "select_sql", lambda dialect, plan, as_of: f'SELECT * FROM "{plan.table.name}"'
    )
    with pytest.raises(SnapshotError, match="event time after"):
        take(demo, workspace, tables=["orders"])
    leftovers = workspace.source.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'snapshots'"
    )
    assert leftovers == []  # a refused snapshot leaves nothing behind


def test_a_table_over_the_row_limit_fails_loudly_and_leaves_nothing(
    demo: Demo, workspace: Workspace
) -> None:
    with pytest.raises(SnapshotError, match="more than 100 rows"):
        take(demo, workspace, limits=SnapshotLimits(row_limit_per_table=100))
    assert (
        workspace.source.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'snapshots'"
        )
        == []
    )


def test_unknown_tables_and_columns_are_refused(demo: Demo, workspace: Workspace) -> None:
    with pytest.raises(SnapshotError, match="No table 'nope'"):
        take(demo, workspace, tables=["nope"])
    with pytest.raises(SnapshotError, match="no column"):
        take(demo, workspace, tables=["orders"], columns={"orders": ["order_id", "nope"]})


def test_a_snapshot_cannot_name_a_table_outside_the_schema_graph_by_sql_injection(
    demo: Demo, workspace: Workspace
) -> None:
    with pytest.raises(SnapshotError):
        take(demo, workspace, tables=['orders"; DROP TABLE orders; --'])


def test_a_table_without_a_time_column_is_copied_whole(demo: Demo, workspace: Workspace) -> None:
    info = take(demo, workspace, tables=["products"])
    record = info.description.tables["products"]
    assert record.time_column is None and record.max_event_time is None and record.rows > 0


def test_a_numeric_event_time_column_is_refused_with_advice(tmp_path: Path) -> None:
    path = tmp_path / "epoch.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, at INTEGER, v REAL)")
        con.execute("INSERT INTO events VALUES (1, 1700000000, 1.0)")
    demo = Demo("sqlite", path)
    graph = demo.graph()
    table = graph.tables[0]
    table.time_column = "at"
    with pytest.raises(SnapshotError, match="timestamp or date column"):
        plan_tables(graph, "sqlite", None)


# -- acceptance (c): live mode -------------------------------------------------------------


def test_live_mode_records_as_of_counts_and_max_event_times(demo: Demo) -> None:
    info = live_version(
        demo.source(),
        demo.graph(),
        tables=TABLES,
        as_of=AS_OF,
        connection={"id": "c1", "name": "shop"},
    )
    d = info.description
    assert d.mode == "live" and not d.reproducible and d.as_of == AS_OF
    assert "cannot be reproduced exactly" in (d.note or "")
    assert d.connection == {"id": "c1", "name": "shop"}
    assert d.tables["orders"].rows > 1000
    assert str(d.tables["orders"].max_event_time) <= "2024-06-30 12:00:00"
    assert d.tables["customers"].max_event_time is not None
    assert d.n_rows == sum(t.rows for t in d.tables.values())
    print(
        f"\n{demo.dialect} live: as_of={d.as_of.isoformat()} rows={d.n_rows} max={d.max_event_times}"
    )


def test_a_live_version_changes_when_the_data_does_and_never_copies_anything(
    demo: Demo, workspace: Workspace
) -> None:
    a = live_version(demo.source(), demo.graph(), tables=["orders"], as_of=AS_OF)
    demo.add_order(990_300, "2024-06-01 00:00:00")
    b = live_version(demo.source(), demo.graph(), tables=["orders"], as_of=AS_OF)
    assert b.id != a.id
    assert b.description.tables["orders"].rows == a.description.tables["orders"].rows + 1
    with pytest.raises(EngineError):
        workspace.snapshot_tables(a.id)  # nothing was stored


def test_live_mode_ignores_rows_after_as_of_in_its_counts(demo: Demo) -> None:
    a = live_version(demo.source(), demo.graph(), tables=["orders"], as_of=AS_OF)
    demo.add_order(990_400, FUTURE)
    b = live_version(demo.source(), demo.graph(), tables=["orders"], as_of=AS_OF)
    assert b.description.tables["orders"] == a.description.tables["orders"]


def test_a_naive_as_of_is_taken_as_utc(demo: Demo, workspace: Workspace) -> None:
    naive = create_snapshot(
        demo.source(),
        demo.graph(),
        workspace,
        tables=["orders"],
        as_of=datetime(2024, 6, 30, 12),  # noqa: DTZ001 (naive on purpose)
    )
    aware = take(demo, workspace, tables=["orders"])
    assert naive.id == aware.id and naive.description.as_of == AS_OF
