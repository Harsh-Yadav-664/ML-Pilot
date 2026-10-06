"""The schema graph on a real Postgres (#45): declared keys, inferred keys, times, row counts."""

from __future__ import annotations

from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import open_source
from tests.fixtures.demo_db import demo_module, load_demo_postgres
from tests.integration.conftest import SCHEMA, admin_connection


def _graph(spec) -> SchemaGraph:
    return build_schema_graph(open_source(spec))


def _edges(graph: SchemaGraph) -> set[tuple[str, str, str, str]]:
    prefix = f"{SCHEMA}."
    return {
        (
            e.from_table.removeprefix(prefix),
            e.from_columns[0],
            e.to_table.removeprefix(prefix),
            e.to_columns[0],
        )
        for e in graph.edges
        if e.from_table.startswith(prefix)
    }


def _expected() -> set[tuple[str, str, str, str]]:
    edges = set()
    for table in demo_module().SCHEMA:
        for column in table.columns:
            if column.fk:
                parent, parent_column = column.fk.split(".")
                edges.add((table.name, column.name, parent, parent_column))
    return edges


def test_declared_keys_times_and_row_counts_in_postgres(demo_pg) -> None:
    _, ro, _ = demo_pg
    graph = _graph(ro)
    assert _edges(graph) == _expected() and len(_edges(graph)) == 9
    assert {e.source for e in graph.edges if e.from_table.startswith(SCHEMA)} == {"declared"}
    tables = {t.name: t for t in graph.tables if t.db_schema == SCHEMA}
    assert set(tables) == {t.name for t in demo_module().SCHEMA}
    assert tables["customers"].row_count == 800 and not tables["customers"].row_count_estimated
    assert tables["orders"].primary_key == ["order_id"]
    for name, column in {
        "orders": "ordered_at",
        "sessions": "started_at",
        "support_tickets": "opened_at",
        "refunds": "refunded_at",
        "marketing_emails": "sent_at",
    }.items():
        assert tables[name].time_column == column
    assert [t.name for t in tables.values() if t.time_leakage_hint] == ["customer_status_snapshot"]
    assert tables["customers"].key == f"{SCHEMA}.customers"  # a non-default schema is qualified


def test_inferred_keys_in_postgres_when_the_foreign_keys_are_dropped(demo_pg) -> None:
    admin_spec, ro, _ = demo_pg
    admin = admin_connection(admin_spec)
    try:
        load_demo_postgres(admin, SCHEMA, fks=False)
        admin.run(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {ro.username}")
        admin.run(f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO {ro.username}")
    finally:
        admin.close()
    graph = _graph(ro)
    found, expected = _edges(graph), _expected()
    precision = len(found & expected) / len(found)
    recall = len(found & expected) / len(expected)
    print(f"\nPostgres without foreign keys: precision {precision:.2f}, recall {recall:.2f}")
    assert precision >= 0.9 and recall >= 0.9
    assert {e.source for e in graph.edges if e.from_table.startswith(SCHEMA)} == {"inferred"}
