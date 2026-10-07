"""Columns that are overwritten after their row's event time (#139).

The point-in-time guard bounds rows by their event time; it cannot see that ``is_churned`` or
``orders.status`` is filled in or changed later. Names that look like a status or flag give a
warning; the others are found by comparing two snapshots of the database, and a query that reads
one then gets the same warning, naming the column.
"""

# ruff: noqa: F811
from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import pytest

from ml.data.engine import arrow_to_pandas
from ml.data.schema_graph import SchemaGraph, SchemaOverrides, apply_overrides, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.mutable_columns import observe_mutable
from ml.tasks.pit_guard import check
from ml.tasks.pit_verify import tables_from_source
from tests.fixtures.api import API
from tests.fixtures.demo_db import demo_duckdb
from tests.integration.test_baseline_run_api import connection  # noqa: F401
from tests.integration.test_canaries import IS_CHURNED
from tests.integration.test_tasks_api import take_snapshot

MAX_STATUS = """
SELECT l.entity_id, l.cutoff_time, MAX(o.status) AS value
FROM __labels l JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
"""
SUM_TOTAL = """
SELECT l.entity_id, l.cutoff_time, SUM(o.total) AS value
FROM __labels l JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
"""
COUNT_ORDERS = """
SELECT l.entity_id, l.cutoff_time, COUNT(*) AS value
FROM __labels l JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
"""
READ_X1 = """
SELECT l.entity_id, l.cutoff_time, CAST(c.x1 AS INTEGER) AS value
FROM __labels l JOIN customers c ON c.customer_id = l.entity_id
"""
READ_X2 = """
SELECT l.entity_id, l.cutoff_time,
       CASE WHEN c.x2 IS NOT NULL THEN 1 ELSE 0 END AS value
FROM __labels l JOIN customers c ON c.customer_id = l.entity_id
"""


def mutable_warnings(sql: str, graph: SchemaGraph) -> list[str]:
    result = check(sql, graph)
    assert result.status in ("accepted", "rewritten"), result.reasons
    return [w for w in result.warnings if "may be overwritten" in w]


def graph_of(path: Path) -> tuple[SchemaGraph, object]:
    source = open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
    return build_schema_graph(source), source


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return demo_duckdb(tmp_path_factory.mktemp("mutable") / "demo.duckdb")


def test_a_query_reading_is_churned_or_an_order_status_warns_and_names_the_column(
    demo: Path,
) -> None:
    graph, _ = graph_of(demo)
    churned = mutable_warnings(IS_CHURNED, graph)
    status = mutable_warnings(MAX_STATUS, graph)
    assert len(churned) == 1 and "customers.is_churned" in churned[0]
    assert len(status) == 1 and "orders.status" in status[0]
    print(f"\nis_churned: {churned[0][:150]}...\nMAX(orders.status): {status[0][:150]}...")


def test_immutable_columns_of_the_demo_database_get_no_warning(demo: Path) -> None:
    graph, _ = graph_of(demo)
    assert mutable_warnings(SUM_TOTAL, graph) == []  # orders.total
    assert mutable_warnings(COUNT_ORDERS, graph) == []  # orders.customer_id, ordered_at
    flagged = sorted(f"{t.name}.{c.name}" for t in graph.tables for c in t.columns if c.mutable)
    assert "orders.total" not in flagged and "orders.customer_id" not in flagged
    print(f"\ncolumns flagged by name on the demo database: {flagged}")


def test_confirming_a_column_as_immutable_silences_the_warning(demo: Path) -> None:
    graph, _ = graph_of(demo)
    confirmed = apply_overrides(graph, SchemaOverrides(immutable_columns={"orders": ["status"]}))
    assert mutable_warnings(MAX_STATUS, confirmed) == []
    assert mutable_warnings(IS_CHURNED, confirmed) != []  # only the confirmed column


def test_a_column_with_an_innocent_name_is_found_by_comparing_two_snapshots(demo: Path) -> None:
    """The canaries 4 and 7 columns under names no name check would flag."""
    original, source = graph_of(demo)
    names = {"is_churned": "x1", "discount_code_used_after_churn": "x2"}
    graph = original.model_copy(deep=True)  # the same schema with the two columns renamed
    for table in graph.tables:
        for column in table.columns:
            if table.name == "customers" and column.name in names:
                column.name, column.mutable, column.mutable_source = names[column.name], None, None
    # by name alone nothing is flagged
    assert mutable_warnings(READ_X1, graph) == [] and mutable_warnings(READ_X2, graph) == []

    newer = {
        k: arrow_to_pandas(v)
        for k, v in tables_from_source(source, ["customers", "orders"]).items()
    }
    newer["customers"] = newer["customers"].rename(columns=names)
    older = {k: v.copy() for k, v in newer.items()}
    # when the rows were first written the two columns were still empty
    older["customers"]["x1"] = False
    older["customers"]["x2"] = None
    report = observe_mutable(older, newer, graph)
    found = {(o.table, o.column): o for o in report.mutable}
    assert set(found) == {("customers", "x1"), ("customers", "x2")}, report.mutable
    assert found[("customers", "x1")].compared_rows == len(newer["customers"])

    flagged = apply_overrides(graph, SchemaOverrides(mutable_columns=report.as_overrides()))
    w1, w2 = mutable_warnings(READ_X1, flagged), mutable_warnings(READ_X2, flagged)
    assert len(w1) == 1 and "customers.x1" in w1[0]
    assert len(w2) == 1 and "customers.x2" in w2[0]
    # an unchanged table is not flagged: only what changed between the snapshots is seen
    assert mutable_warnings(MAX_STATUS, flagged) != []  # still by name
    assert mutable_warnings(SUM_TOTAL, flagged) == []
    print(
        f"\nrenamed to x1/x2: no warning by name; after comparing two snapshots: "
        f"{w1[0][:80]}... / {w2[0][:80]}..."
    )


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_comparing_snapshots_handles_missing_values_keys_and_absent_tables(demo: Path) -> None:
    graph, _ = graph_of(demo)
    orders = pd.DataFrame(
        {"order_id": [1, 2, 3], "status": ["paid", None, "paid"], "total": [1.0, 2.0, 3.0]}
    )
    changed = orders.copy()
    changed.loc[2, "status"] = "refunded"  # a real change
    same = observe_mutable({"orders": orders}, {"orders": orders.copy()}, graph)
    assert same.mutable == []  # a missing value that stays missing is not a change
    diff = observe_mutable({"orders": orders}, {"orders": changed}, graph)
    assert [(o.table, o.column, o.changed_rows) for o in diff.mutable] == [("orders", "status", 1)]
    # rows that exist in only one snapshot are not compared
    grown = pd.concat([orders, frame([{"order_id": 9, "status": "x", "total": 9.0}])])
    assert observe_mutable({"orders": orders}, {"orders": grown}, graph).mutable == []
    # a table in only one snapshot, or without a unique key, is listed, not ignored
    dup = pd.concat([orders, orders])
    out = observe_mutable({"orders": dup}, {"orders": orders}, graph)
    assert ("orders", "its primary key is not unique in the data") in out.skipped
    assert ("customers", "not in both snapshots") in out.skipped


async def test_the_mutable_check_endpoint_compares_two_snapshots_and_saves_the_result(
    client: httpx.AsyncClient, project_id: str, connection: str, tmp_path: Path
) -> None:
    resp = await client.get(f"{API}/projects/{project_id}/connections/{connection}")
    path = Path(resp.json()["database"])
    first = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    with sqlite3.connect(path) as db:  # the shop corrects 40 old orders after the first snapshot
        db.execute(
            "UPDATE orders SET total = total + 1 WHERE order_id IN "
            "(SELECT order_id FROM orders ORDER BY order_id LIMIT 40)"
        )
    second = await take_snapshot(client, project_id, connection, "2025-01-02T00:00:00Z")
    assert first != second

    url = f"{API}/projects/{project_id}/connections/{connection}/schema"
    same = await client.post(
        f"{url}/mutable-check", json={"older_version_id": first, "newer_version_id": first}
    )
    assert same.status_code == 422
    out = await client.post(
        f"{url}/mutable-check", json={"older_version_id": first, "newer_version_id": second}
    )
    assert out.status_code == 200, out.text
    body = out.json()
    assert [(m["table"], m["column"], m["changed_rows"]) for m in body["mutable"]] == [
        ("orders", "total", 40)
    ]
    assert body["saved"] is True and body["checked_tables"] == 2
    assert "not proven immutable" in body["note"]

    schema = (await client.get(url)).json()
    orders = next(t for t in schema["tables"] if t["name"] == "orders")
    total = next(c for c in orders["columns"] if c["name"] == "total")
    assert total["mutable_source"] == "observed" and total["mutable"]
    assert schema["overrides"]["mutable_columns"] == {"orders": ["total"]}
    # the user can confirm it as immutable again
    patched = await client.patch(url, json={"immutable_columns": {"orders": ["total"]}})
    assert patched.status_code == 200
    total = next(
        c
        for t in patched.json()["tables"]
        if t["name"] == "orders"
        for c in t["columns"]
        if c["name"] == "total"
    )
    assert total["mutable"] is None and total["mutable_source"] == "user"
    print(f"\nHTTP 200: {body['mutable']}; saved; then confirmed immutable by the user")
