"""Snapshots of a real Postgres (#97): the cutoff, the id, and timestamp with time zone columns."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ml.data.engine import DuckDBSource
from ml.data.schema_graph import build_schema_graph
from ml.data.snapshot import create_snapshot, live_version
from ml.data.sources import open_source
from ml.data.workspace import Workspace
from tests.integration.conftest import SCHEMA, admin_connection

AS_OF = datetime(2024, 6, 30, 12, 0, 0, tzinfo=UTC)
ORDERS = f"{SCHEMA}.orders"


def test_future_rows_are_in_postgres_but_not_in_the_snapshot_and_the_id_follows_the_data(
    demo_pg, tmp_path: Path
) -> None:
    admin, ro, _ = demo_pg
    source = open_source(ro)
    graph = build_schema_graph(source)
    workspace = Workspace(DuckDBSource(tmp_path / "work.duckdb"))
    try:
        keys = [ORDERS, f"{SCHEMA}.customers"]
        clean = create_snapshot(source, graph, workspace, tables=keys, as_of=AS_OF)
        again = create_snapshot(source, graph, workspace, tables=keys, as_of=AS_OF)
        assert clean.id == again.id and clean.created and not again.created

        db = admin_connection(admin)
        try:
            for i, when in enumerate(["2999-01-01 00:00:00", "2024-06-30 12:00:01"]):
                db.run(
                    f"INSERT INTO {SCHEMA}.orders VALUES (:i, 1, :t, 'completed', 9.99)",
                    i=990_000 + i,
                    t=datetime.fromisoformat(when),
                )
            assert db.run(f"SELECT count(*) FROM {SCHEMA}.orders WHERE order_id >= 990000") == [[2]]
            after_canaries = create_snapshot(source, graph, workspace, tables=keys, as_of=AS_OF)
            assert after_canaries.id == clean.id  # the canaries are not part of the data
            ref = workspace.snapshot_tables(after_canaries.id)[ORDERS]
            assert workspace.source.execute(
                f"SELECT count(*) FROM {ref.sql()} WHERE order_id >= 990000"
            ) == [(0,)]

            db.run(
                f"INSERT INTO {SCHEMA}.orders VALUES (991000, 1, '2024-06-01 10:00:00', 'completed', 9.99)"
            )
            changed = create_snapshot(source, graph, workspace, tables=keys, as_of=AS_OF)
            assert changed.id != clean.id
            print(f"\npostgres: {clean.short_hash} == {again.short_hash} != {changed.short_hash}")
        finally:
            db.close()
    finally:
        workspace.source.close()


def test_a_timestamptz_event_time_is_compared_as_utc(demo_pg, tmp_path: Path) -> None:
    admin, ro, _ = demo_pg
    db = admin_connection(admin)
    try:
        db.run(f"CREATE TABLE {SCHEMA}.events (id int PRIMARY KEY, at timestamptz NOT NULL, v int)")
        # 11:30 UTC (13:30 +02:00) is before the cutoff; 12:30 UTC (14:30 +02:00) is after it.
        db.run(
            f"INSERT INTO {SCHEMA}.events VALUES "
            "(1, '2024-06-30 13:30:00+02', 1), (2, '2024-06-30 14:30:00+02', 2), "
            "(3, '2024-06-30 12:00:00+00', 3)"
        )
        db.run(f"GRANT SELECT ON {SCHEMA}.events TO {ro.username}")
    finally:
        db.close()
    source = open_source(ro)
    graph = build_schema_graph(source)
    table = next(t for t in graph.tables if t.key == f"{SCHEMA}.events")
    assert table.time_column == "at"
    workspace = Workspace(DuckDBSource(tmp_path / "work.duckdb"))
    try:
        info = create_snapshot(source, graph, workspace, tables=[table.key], as_of=AS_OF)
        ref = workspace.snapshot_tables(info.id)[table.key]
        ids = sorted(r[0] for r in workspace.source.execute(f"SELECT id FROM {ref.sql()}"))
        assert ids == [1, 3]  # id 3 is exactly at as_of, id 2 is an hour and a half later in UTC
        assert info.description.tables[table.key].max_event_time == "2024-06-30 12:00:00"
        live = live_version(source, graph, tables=[table.key], as_of=AS_OF)
        assert live.description.tables[table.key].rows == 2
    finally:
        workspace.source.close()
