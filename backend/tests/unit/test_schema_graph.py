"""The schema graph (#45): declared and inferred keys, time columns, leakage hint, overrides.

The demo database (#47) is read as a SQLite and a DuckDB file, with and without its declared
foreign keys. A second, deliberately messy database checks what the demo's tidy names cannot:
role-named keys, irregular plurals, text keys, and look-alike columns that must not become edges.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ml.data.schema_graph import (
    Column,
    EdgeRef,
    OverrideError,
    SchemaGraph,
    SchemaOverrides,
    apply_overrides,
    build_schema_graph,
    id_stem,
    merge_overrides,
    name_strength,
    singular,
    time_candidates,
    type_kind,
    validate_overrides,
)
from ml.data.sources import ConnectionSpec, open_source
from tests.fixtures.demo_db import demo_duckdb, demo_module, demo_sqlite

EVENT_TABLES = {
    "orders": "ordered_at",
    "sessions": "started_at",
    "support_tickets": "opened_at",
    "refunds": "refunded_at",
    "marketing_emails": "sent_at",
}


def declared_in_generator() -> set[tuple[str, str, str, str]]:
    """The demo database's foreign keys as (table, column, parent table, parent column)."""
    edges = set()
    for table in demo_module().SCHEMA:
        for column in table.columns:
            if column.fk:
                parent, parent_column = column.fk.split(".")
                edges.add((table.name, column.name, parent, parent_column))
    return edges


def edge_set(graph: SchemaGraph, *sources: str) -> set[tuple[str, str, str, str]]:
    return {
        (e.from_table, e.from_columns[0], e.to_table, e.to_columns[0])
        for e in graph.edges
        if not sources or e.source in sources
    }


def read(path: Path, dialect: str, overrides: SchemaOverrides | None = None) -> SchemaGraph:
    source = open_source(ConnectionSpec(dialect=dialect, database=str(path)))  # type: ignore[arg-type]
    return build_schema_graph(source, overrides)


def demo_graph(tmp_path: Path, dialect: str, *, fks: bool) -> SchemaGraph:
    make = demo_sqlite if dialect == "sqlite" else demo_duckdb
    return read(make(tmp_path / f"demo.{dialect}", fks=fks), dialect)


# -- the demo database -----------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_every_declared_foreign_key_of_the_demo_database_is_found(
    tmp_path: Path, dialect: str
) -> None:
    graph = demo_graph(tmp_path, dialect, fks=True)
    expected = declared_in_generator()
    assert len(expected) == 9
    assert edge_set(graph) == expected
    assert all(e.source == "declared" and e.confidence == 1.0 for e in graph.edges)
    assert not graph.warnings
    print(f"\n{dialect}: {len(graph.edges)} of {len(expected)} declared foreign keys found")


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_inferred_keys_on_the_demo_database_without_declared_foreign_keys(
    tmp_path: Path, dialect: str
) -> None:
    graph = demo_graph(tmp_path, dialect, fks=False)
    expected = declared_in_generator()
    found = edge_set(graph)
    precision = len(found & expected) / len(found)
    recall = len(found & expected) / len(expected)
    print(f"\n{dialect} without foreign keys: precision {precision:.2f}, recall {recall:.2f}")
    assert precision >= 0.9 and recall >= 0.9
    assert all(e.source == "inferred" for e in graph.edges)
    assert all(e.overlap is not None and e.overlap >= 0.95 for e in graph.edges)
    assert all(0.9 < e.confidence <= 1.0 for e in graph.edges)


def test_cardinality_follows_the_primary_key(tmp_path: Path) -> None:
    for fks in (True, False):
        folder = tmp_path / f"fks{fks}"
        folder.mkdir()
        graph = demo_graph(folder, "sqlite", fks=fks)
        one_to_one = {(e.from_table, e.to_table) for e in graph.edges if e.cardinality == "1:1"}
        # The status snapshot has one row per customer: its key is the customer id.
        assert one_to_one == {("customer_status_snapshot", "customers")}


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_every_event_table_has_its_time_column_and_only_the_snapshot_is_a_leakage_hint(
    tmp_path: Path, dialect: str
) -> None:
    graph = demo_graph(tmp_path, dialect, fks=True)
    tables = {t.key: t for t in graph.tables}
    for name, column in EVENT_TABLES.items():
        assert tables[name].time_column == column, (name, tables[name].time_candidates)
        assert tables[name].time_column_source == "inferred"
        assert tables[name].time_leakage_hint is None
    # An opened ticket beats its resolution time.
    assert tables["support_tickets"].time_candidates == ["opened_at", "resolved_at"]
    # The snapshot is rewritten after churn and has no other time column.
    snapshot = tables["customer_status_snapshot"]
    assert snapshot.time_column == "updated_at"
    assert snapshot.time_leakage_hint and "last-modified" in snapshot.time_leakage_hint
    assert [t.key for t in graph.tables if t.time_leakage_hint] == ["customer_status_snapshot"]
    # Tables without any time column are not given one.
    assert tables["products"].time_column is None and tables["order_items"].time_column is None
    assert tables["customers"].time_column == "signup_at"
    assert all(t.is_static is None for t in graph.tables)  # only a user can say "static"


def test_row_counts_primary_keys_and_hints_of_the_demo_tables(tmp_path: Path) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=True)
    tables = {t.key: t for t in graph.tables}
    assert tables["customers"].row_count == 800 and not tables["customers"].row_count_estimated
    assert tables["orders"].primary_key == ["order_id"]
    hints = {c.name: c.hint for c in tables["orders"].columns}
    assert hints == {
        "order_id": "id",
        "customer_id": "id",
        "ordered_at": "time",
        "status": "text",
        "total": "numeric",
    }
    customers = {c.name: c.hint for c in tables["customers"].columns}
    assert customers["is_churned"] == "boolean" and customers["signup_at"] == "time"
    assert tables["customers"].columns[0].is_primary_key


def test_the_graph_is_json_serialisable_and_round_trips(tmp_path: Path) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=False)
    assert SchemaGraph.model_validate_json(graph.model_dump_json()) == graph


# -- a messy database ------------------------------------------------------------------


@pytest.fixture
def messy(tmp_path: Path) -> Path:
    path = tmp_path / "messy.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, joined TIMESTAMP);
        CREATE TABLE addresses (id INTEGER PRIMARY KEY, street TEXT);
        CREATE TABLE categories (category_id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE products (sku TEXT PRIMARY KEY, category_id INTEGER, title TEXT);
        -- no primary key at all, but unique ids
        CREATE TABLE warehouses (warehouse_id INTEGER, city TEXT);
        CREATE TABLE orders (
            order_no INTEGER PRIMARY KEY,
            customer_id INTEGER,
            billing_address_id INTEGER,
            shipping_address_id INTEGER,
            product_key TEXT,
            warehouse_id INTEGER,
            region_id INTEGER,
            legacy_customer_id INTEGER,
            half_customer_id INTEGER,
            created TEXT,
            created_at_epoch INTEGER,
            updated_at TIMESTAMP
        );
        -- a declared key with no column list: it means the parent's primary key
        CREATE TABLE notes (
            note_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers
        );
        -- a declared composite key
        CREATE TABLE stock (
            warehouse_id INTEGER, sku TEXT, qty INTEGER,
            PRIMARY KEY (warehouse_id, sku),
            FOREIGN KEY (sku) REFERENCES products (sku)
        );
        """
    )
    con.executemany(
        "INSERT INTO customers VALUES (?, ?, ?)",
        [(i, f"c{i}", "2024-01-01") for i in range(1, 101)],
    )
    con.executemany("INSERT INTO addresses VALUES (?, ?)", [(i, f"s{i}") for i in range(1, 51)])
    con.executemany("INSERT INTO categories VALUES (?, ?)", [(i, f"k{i}") for i in range(1, 6)])
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [(f"sku-{i}", i % 5 + 1, f"t{i}") for i in range(30)],
    )
    con.executemany("INSERT INTO warehouses VALUES (?, ?)", [(i, f"city{i}") for i in range(1, 4)])
    rows = []
    for i in range(1, 301):
        rows.append(
            (
                i,
                i % 100 + 1,
                i % 50 + 1,
                (i * 7) % 50 + 1,
                f"sku-{i % 30}",
                i % 3 + 1,
                i % 9 + 1,  # there is no regions table
                1000 + i,  # ids that are not in customers at all
                i % 200 + 1,  # only half of them exist in customers (100 rows)
                "2024-02-01",
                1_700_000_000 + i,
                "2024-03-01",
            )
        )
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO notes VALUES (?, ?)", [(i, i) for i in range(1, 20)])
    con.executemany("INSERT INTO stock VALUES (?, ?, ?)", [(1, f"sku-{i}", i) for i in range(10)])
    con.commit()
    con.close()
    return path


def test_a_messy_database_gets_exactly_the_right_edges(messy: Path) -> None:
    graph = read(messy, "sqlite")
    assert edge_set(graph, "declared") == {
        ("notes", "customer_id", "customers", "id"),
        ("stock", "sku", "products", "sku"),
    }
    assert edge_set(graph, "inferred") == {
        ("orders", "customer_id", "customers", "id"),
        ("orders", "billing_address_id", "addresses", "id"),  # a role name
        ("orders", "shipping_address_id", "addresses", "id"),
        ("orders", "product_key", "products", "sku"),  # a text key
        ("orders", "warehouse_id", "warehouses", "warehouse_id"),  # parent without a primary key
        ("products", "category_id", "categories", "category_id"),  # irregular plural
        ("stock", "warehouse_id", "warehouses", "warehouse_id"),
    }
    # Not edges: no regions table; ids absent from customers; ids only half present.
    assert not [e for e in graph.edges if e.from_columns[0] in ("region_id", "legacy_customer_id")]
    assert not [e for e in graph.edges if e.from_columns[0] == "half_customer_id"]
    role = next(e for e in graph.edges if e.from_columns == ["billing_address_id"])
    exact = next(
        e for e in graph.edges if e.from_columns == ["customer_id"] and e.from_table == "orders"
    )
    assert role.confidence < exact.confidence  # a role name is weaker evidence


def test_a_declared_key_is_not_inferred_a_second_time(messy: Path) -> None:
    graph = read(messy, "sqlite")
    stock = {(e.source, e.from_columns[0]) for e in graph.edges if e.from_table == "stock"}
    # sku is declared; warehouse_id is part of the primary key but not declared, so it is inferred.
    assert stock == {("declared", "sku"), ("inferred", "warehouse_id")}
    assert (
        len([e for e in graph.edges if e.from_table == "stock" and e.from_columns == ["sku"]]) == 1
    )


def test_time_columns_of_the_messy_database(messy: Path) -> None:
    tables = {t.key: t for t in read(messy, "sqlite").tables}
    # A typed timestamp beats text and epoch columns that only look like times by name.
    assert tables["orders"].time_column == "created"
    assert tables["orders"].time_candidates[0] == "created"
    assert set(tables["orders"].time_candidates) == {"created", "created_at_epoch", "updated_at"}
    assert tables["orders"].time_candidates[-1] == "updated_at"
    assert tables["customers"].time_column == "joined"
    assert tables["warehouses"].time_column is None


def test_the_check_budget_is_reported_not_hidden(messy: Path) -> None:
    source = open_source(ConnectionSpec(dialect="sqlite", database=str(messy)))
    graph = build_schema_graph(source, max_overlap_checks=2)
    assert any("Inference stopped after 2" in w for w in graph.warnings)


# -- small pieces ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_name", "kind"),
    [
        ("bigint", "integer"),
        ("INTEGER", "integer"),
        ("double precision", "float"),
        ("DECIMAL(10,2)", "float"),
        ("character varying", "text"),
        ("VARCHAR", "text"),
        ("timestamp without time zone", "timestamp"),
        ("TIMESTAMP WITH TIME ZONE", "timestamp"),
        ("datetime", "timestamp"),
        ("date", "date"),
        ("boolean", "boolean"),
        ("jsonb", "other"),
    ],
)
def test_type_kind(type_name: str, kind: str) -> None:
    assert type_kind(type_name) == kind


@pytest.mark.parametrize(
    ("column", "stem"),
    [
        ("customer_id", "customer"),
        ("CustomerID", "customer"),
        ("customerid", None),
        ("product_sku", None),
        ("billing_address_id", "billing_address"),
        ("order_key", "order"),
        ("id", None),
        ("_id", None),
        ("paid", None),
    ],
)
def test_id_stem(column: str, stem: str | None) -> None:
    assert id_stem(column) == stem


def test_the_loose_form_is_only_for_inference() -> None:
    assert id_stem("customerid", loose=True) == "customer"
    assert id_stem("paid", loose=True) is None and id_stem("valid", loose=True) == "val"


def test_singular_and_name_strength() -> None:
    assert [singular(w) for w in ("customers", "categories", "addresses", "status", "boxes")] == [
        "customer",
        "category",
        "address",
        "status",
        "box",
    ]
    assert name_strength("customer", "customers") == 1.0
    assert name_strength("category", "categories") == 1.0
    assert name_strength("billing_address", "addresses") == 0.7
    assert name_strength("customer", "shop_customers") == 0.7
    assert name_strength("customer", "orders") == 0.0


def col(name: str, type_name: str) -> Column:
    return Column(name=name, type=type_name, nullable=True, hint="other")


def test_time_candidates_rank_creation_over_end_over_attribute_over_update() -> None:
    columns = [
        col("updated_at", "timestamp"),
        col("date_of_birth", "date"),
        col("resolved_at", "timestamp"),
        col("created_at", "timestamp"),
        col("notes", "text"),
        col("amount", "double"),
    ]
    assert [name for _, name in time_candidates(columns)] == [
        "created_at",
        "resolved_at",
        "date_of_birth",
        "updated_at",
    ]
    assert time_candidates([col("event_time", "text")])[0][1] == "event_time"
    assert time_candidates([col("name", "text"), col("n", "integer")]) == []


# -- overrides -------------------------------------------------------------------------


def test_overrides_change_time_column_static_flag_and_edges(tmp_path: Path) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=True)
    drop = EdgeRef(
        from_table="refunds",
        from_columns=["customer_id"],
        to_table="customers",
        to_columns=["customer_id"],
    )
    add = EdgeRef(
        from_table="sessions",
        from_columns=["session_id"],
        to_table="orders",
        to_columns=["order_id"],
    )
    overrides = SchemaOverrides(
        time_columns={"support_tickets": "resolved_at", "orders": None},
        static_tables={"products": True},
        add_edges=[add],
        remove_edges=[drop],
    )
    validate_overrides(graph, overrides)
    changed = apply_overrides(graph, overrides)
    tables = {t.key: t for t in changed.tables}
    assert tables["support_tickets"].time_column == "resolved_at"
    assert tables["support_tickets"].time_column_source == "user"
    assert tables["orders"].time_column is None and tables["orders"].time_column_source == "user"
    assert tables["products"].is_static is True
    assert ("refunds", "customer_id", "customers", "customer_id") not in edge_set(changed)
    added = next(e for e in changed.edges if add.matches(e))
    assert added.source == "user" and added.confidence == 1.0
    assert changed.overrides == overrides
    assert len(changed.edges) == len(graph.edges)  # one removed, one added
    # The input graph is not modified.
    assert {t.key: t for t in graph.tables}["orders"].time_column == "ordered_at"


def test_choosing_an_update_time_as_the_time_column_keeps_the_leakage_hint(tmp_path: Path) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=True)
    # customer_status_snapshot only has updated_at: the user can confirm it, the hint stays.
    changed = apply_overrides(
        graph, SchemaOverrides(time_columns={"customer_status_snapshot": "updated_at"})
    )
    snapshot = {t.key: t for t in changed.tables}["customer_status_snapshot"]
    assert snapshot.time_column_source == "user" and snapshot.time_leakage_hint


@pytest.mark.parametrize(
    "overrides",
    [
        SchemaOverrides(time_columns={"nope": "x"}),
        SchemaOverrides(time_columns={"orders": "not_a_column"}),
        SchemaOverrides(static_tables={"nope": True}),
        SchemaOverrides(
            add_edges=[
                EdgeRef(
                    from_table="orders",
                    from_columns=["customer_id"],
                    to_table="nope",
                    to_columns=["x"],
                )
            ]
        ),
        SchemaOverrides(
            add_edges=[
                EdgeRef(
                    from_table="orders",
                    from_columns=["customer_id", "order_id"],
                    to_table="customers",
                    to_columns=["customer_id"],
                )
            ]
        ),
    ],
)
def test_overrides_naming_something_that_does_not_exist_are_refused(
    tmp_path: Path, overrides: SchemaOverrides
) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=True)
    with pytest.raises(OverrideError):
        validate_overrides(graph, overrides)


def test_saved_overrides_that_no_longer_match_the_database_are_skipped_and_reported(
    tmp_path: Path,
) -> None:
    graph = demo_graph(tmp_path, "sqlite", fks=True)
    stale = SchemaOverrides(
        time_columns={"gone": "x"},
        static_tables={"gone": True},
        add_edges=[
            EdgeRef(
                from_table="gone", from_columns=["a"], to_table="orders", to_columns=["order_id"]
            )
        ],
    )
    changed = apply_overrides(graph, stale)
    assert changed.edges == graph.edges
    assert len(changed.warnings) == 3 and all("gone" in w for w in changed.warnings)


def test_a_patch_replaces_the_fields_it_gives_and_keeps_the_others() -> None:
    stored = SchemaOverrides(
        time_columns={"orders": "ordered_at"}, static_tables={"products": True}
    ).model_dump(mode="json")
    merged = merge_overrides(stored, SchemaOverrides(static_tables={}))
    assert merged.time_columns == {"orders": "ordered_at"}
    assert merged.static_tables == {}
    assert merge_overrides(None, SchemaOverrides()) == SchemaOverrides()
