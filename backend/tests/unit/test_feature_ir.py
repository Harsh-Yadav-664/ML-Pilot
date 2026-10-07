"""Feature IR (#100): expected SQL and sentences, field-level errors, and a property test.

* 12 hand-written IRs, each with the exact SQL it must compile to and the sentence ``describe``
  must give. The same IRs also compile for Postgres and pass the point-in-time guard there.
* Invalid IRs (a sum over a text column, a path that is not in the graph, ...) are rejected, and
  each error names the field.
* Property test: 500 random valid IRs on the demo database compile, pass the point-in-time guard
  and give the same values when every row after each cutoff is deleted.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from ml.data.schema_graph import (
    EdgeRef,
    SchemaGraph,
    SchemaOverrides,
    apply_overrides,
    build_schema_graph,
    type_kind,
)
from ml.data.sources import ConnectionSpec, open_source
from ml.features.compile import compile
from ml.features.ir import FeatureIR, FeatureIRError, Predicate, describe, validate
from ml.tasks.pit_guard import check
from ml.tasks.pit_verify import run_feature, tables_from_source, truncation_check
from tests.fixtures.demo_db import demo_duckdb

ALL_TABLES = [
    "customers",
    "orders",
    "order_items",
    "products",
    "refunds",
    "sessions",
    "support_tickets",
    "marketing_emails",
]
FIXED_CUTOFFS = ["2023-06-01", "2023-12-01", "2024-04-01"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    path = demo_duckdb(tmp_path_factory.mktemp("ir") / "demo.duckdb")
    source = open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
    # order_items and products have no event time: the user confirms them as static
    graph = apply_overrides(
        build_schema_graph(source),
        SchemaOverrides(static_tables={"order_items": True, "products": True}),
    )
    tables = tables_from_source(source, ALL_TABLES)
    customers = tables["customers"].to_pandas()
    orders = tables["orders"].to_pandas()
    rng = random.Random(3)
    entities = sorted(rng.sample(sorted(customers.customer_id.tolist()), 80))
    rows = [(e, pd.Timestamp(c)) for e in entities for c in FIXED_CUTOFFS]
    for e in entities:  # one cutoff per entity placed exactly on one of its orders
        mine = orders[orders.customer_id == e].ordered_at
        if len(mine):
            rows.append((e, mine.sort_values().iloc[len(mine) // 2]))
    labels = pd.DataFrame(rows, columns=["entity_id", "cutoff_time"])
    return {"graph": graph, "tables": tables, "labels": labels}


def edge(child: str, parent: str, child_col: str, parent_col: str) -> EdgeRef:
    return EdgeRef(
        from_table=child, from_columns=[child_col], to_table=parent, to_columns=[parent_col]
    )


CUSTOMER_EDGE = {
    t: edge(t, "customers", "customer_id", "customer_id")
    for t in ("orders", "refunds", "sessions", "support_tickets", "marketing_emails")
}
ORDER_EDGE = {
    "order_items": edge("order_items", "orders", "order_id", "order_id"),
    "refunds": edge("refunds", "orders", "order_id", "order_id"),
}
PRODUCT_EDGE = edge("order_items", "products", "product_id", "product_id")


def direct(table: str, agg: str, **kw: Any) -> FeatureIR:
    return FeatureIR(
        name=kw.pop("name", f"{agg}_{table}"),
        entity_table="customers",
        path=[CUSTOMER_EDGE[table]],
        source_table=table,
        agg=agg,  # type: ignore[arg-type]
        **kw,
    )


def via_orders(table: str, agg: str, **kw: Any) -> FeatureIR:
    return FeatureIR(
        name=kw.pop("name", f"{agg}_{table}_via_orders"),
        entity_table="customers",
        path=[CUSTOMER_EDGE["orders"], ORDER_EDGE[table]],
        source_table=table,
        agg=agg,  # type: ignore[arg-type]
        **kw,
    )


def select(value: str, joins: str) -> str:
    return (
        f"SELECT l.entity_id, l.cutoff_time, {value} AS value\nFROM __labels l\n{joins}\n"
        "GROUP BY l.entity_id, l.cutoff_time"
    )


def join(table: str, alias: str, *terms: str) -> str:
    head = f"LEFT JOIN {table} {alias}\n"
    return head + "\n".join(f"  {'ON' if i == 0 else 'AND'} {t}" for i, t in enumerate(terms))


def ratio_sql() -> str:
    def part(days: int) -> str:
        sql = select(
            "count(t1.customer_id)",
            join(
                "refunds",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.refunded_at < l.cutoff_time",
                f"t1.refunded_at >= l.cutoff_time - INTERVAL '{days} days'",
            ),
        )
        return "\n".join("  " + line for line in sql.splitlines())

    return (
        f"WITH a AS (\n{part(30)}\n), b AS (\n{part(365)}\n)\n"
        "SELECT l.entity_id, l.cutoff_time, 1.0 * a.value / NULLIF(b.value, 0) AS value\n"
        "FROM __labels l\n"
        "LEFT JOIN a ON a.entity_id = l.entity_id AND a.cutoff_time = l.cutoff_time\n"
        "LEFT JOIN b ON b.entity_id = l.entity_id AND b.cutoff_time = l.cutoff_time"
    )


CASES: list[tuple[str, FeatureIR, str, str]] = [
    (
        "count in a window",
        direct("refunds", "count", window_days=90),
        select(
            "count(t1.customer_id)",
            join(
                "refunds",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.refunded_at < l.cutoff_time",
                "t1.refunded_at >= l.cutoff_time - INTERVAL '90 days'",
            ),
        ),
        "Number of refunds in the 90 days before the cutoff",
    ),
    (
        "count over all history",
        direct("sessions", "count"),
        select(
            "count(t1.customer_id)",
            join("sessions", "t1", "t1.customer_id = l.entity_id", "t1.started_at < l.cutoff_time"),
        ),
        "Number of sessions before the cutoff",
    ),
    (
        "sum of a column",
        direct("refunds", "sum", column="amount", window_days=365),
        select(
            "coalesce(sum(t1.amount), 0)",
            join(
                "refunds",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.refunded_at < l.cutoff_time",
                "t1.refunded_at >= l.cutoff_time - INTERVAL '365 days'",
            ),
        ),
        "Total amount of refunds in the 365 days before the cutoff",
    ),
    (
        "mean with a text filter",
        direct(
            "orders",
            "mean",
            column="total",
            window_days=30,
            filter=[Predicate(column="status", op="=", value="completed")],
        ),
        select(
            "avg(t1.total)",
            join(
                "orders",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.ordered_at < l.cutoff_time",
                "t1.ordered_at >= l.cutoff_time - INTERVAL '30 days'",
                "t1.status = 'completed'",
            ),
        ),
        "Average total of orders where status is 'completed' in the 30 days before the cutoff",
    ),
    (
        "max with an in-list",
        direct(
            "sessions",
            "max",
            column="pages",
            filter=[Predicate(column="device", op="in", value=["mobile", "tablet"])],
        ),
        select(
            "max(t1.pages)",
            join(
                "sessions",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.started_at < l.cutoff_time",
                "t1.device IN ('mobile', 'tablet')",
            ),
        ),
        "Largest pages among sessions where device is one of 'mobile', 'tablet' before the cutoff",
    ),
    (
        "standard deviation with two filters",
        direct(
            "orders",
            "std",
            column="total",
            filter=[
                Predicate(column="total", op=">", value=10),
                Predicate(column="status", op="!=", value="cancelled"),
            ],
        ),
        select(
            "stddev_samp(t1.total)",
            join(
                "orders",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.ordered_at < l.cutoff_time",
                "t1.total > 10",
                "t1.status <> 'cancelled'",
            ),
        ),
        (
            "Standard deviation of total across orders where total is above 10 and status is not "
            "'cancelled' before the cutoff"
        ),
    ),
    (
        "days since the last event",
        direct("orders", "days_since_last"),
        select(
            "date_diff('second', CAST(max(t1.ordered_at) AS TIMESTAMP), "
            "CAST(l.cutoff_time AS TIMESTAMP)) / 86400.0",
            join("orders", "t1", "t1.customer_id = l.entity_id", "t1.ordered_at < l.cutoff_time"),
        ),
        "Days since the most recent of the orders before the cutoff",
    ),
    (
        "days since the first event, with an is-null filter",
        direct(
            "support_tickets",
            "days_since_first",
            filter=[Predicate(column="resolved_at", op="is_null")],
        ),
        select(
            "date_diff('second', CAST(min(t1.opened_at) AS TIMESTAMP), "
            "CAST(l.cutoff_time AS TIMESTAMP)) / 86400.0",
            join(
                "support_tickets",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.opened_at < l.cutoff_time",
                "t1.resolved_at IS NULL",
            ),
        ),
        (
            "Days since the earliest of the support tickets where resolved at is missing "
            "before the cutoff"
        ),
    ),
    (
        "share of rows matching a boolean filter",
        direct(
            "marketing_emails",
            "share",
            window_days=90,
            filter=[Predicate(column="opened", op="=", value=True)],
        ),
        select(
            "avg(CASE WHEN t1.opened = TRUE THEN 1.0 ELSE 0.0 END)",
            join(
                "marketing_emails",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.sent_at < l.cutoff_time",
                "t1.sent_at >= l.cutoff_time - INTERVAL '90 days'",
            ),
        ),
        "Share of marketing emails where opened is true in the 90 days before the cutoff",
    ),
    (
        "count distinct two hops away",
        via_orders("order_items", "count_distinct", column="product_id", window_days=30),
        select(
            "count(DISTINCT t2.product_id)",
            join(
                "orders",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.ordered_at < l.cutoff_time",
                "t1.ordered_at >= l.cutoff_time - INTERVAL '30 days'",
            )
            + "\n"
            + join("order_items", "t2", "t2.order_id = t1.order_id"),
        ),
        (
            "Number of distinct product id values among order items, reached through orders, "
            "in the 30 days before the cutoff"
        ),
    ),
    (
        "a timestamp filter and a log transform",
        direct(
            "support_tickets",
            "count",
            transform="log1p",
            filter=[Predicate(column="resolved_at", op=">=", value="2024-01-01")],
        ),
        select(
            "ln(1 + count(t1.customer_id))",
            join(
                "support_tickets",
                "t1",
                "t1.customer_id = l.entity_id",
                "t1.opened_at < l.cutoff_time",
                "t1.resolved_at >= CAST('2024-01-01' AS TIMESTAMP)",
            ),
        ),
        (
            "Log of (1 + value): number of support tickets where resolved at is at least '2024-01-01' "
            "before the cutoff"
        ),
    ),
    (
        "ratio of two windows",
        direct(
            "refunds", "count", window_days=30, ratio_to=direct("refunds", "count", window_days=365)
        ),
        ratio_sql(),
        (
            "Number of refunds in the 30 days before the cutoff, divided by: number of refunds "
            "in the 365 days before the cutoff"
        ),
    ),
]


def test_there_are_at_least_ten_hand_written_cases() -> None:
    assert len(CASES) >= 10
    assert len({c[0] for c in CASES}) == len(CASES)


@pytest.mark.parametrize(
    ("label", "ir", "expected_sql", "sentence"), CASES, ids=[c[0] for c in CASES]
)
def test_the_ir_compiles_to_the_expected_sql_and_reads_as_a_sentence(
    label: str, ir: FeatureIR, expected_sql: str, sentence: str, world: dict[str, Any]
) -> None:
    graph = world["graph"]
    assert validate(ir, graph) == []
    assert compile(ir, graph, "duckdb") == expected_sql
    assert describe(ir) == sentence
    # the compiled query must also be accepted unchanged by the guard, which reads only the SQL
    verdict = check(expected_sql, graph, "duckdb", allow_rewrite=False)
    assert verdict.status == "accepted", verdict.reasons


@pytest.mark.parametrize(
    ("label", "ir", "expected_sql", "sentence"), CASES, ids=[c[0] for c in CASES]
)
def test_the_same_ir_compiles_for_postgres_and_passes_the_guard_there(
    label: str, ir: FeatureIR, expected_sql: str, sentence: str, world: dict[str, Any]
) -> None:
    sql = compile(ir, world["graph"], "postgres")  # compile() runs the guard on its output
    assert "__labels" in sql and "l.cutoff_time" in sql
    if "days_since" in ir.agg:
        assert "EXTRACT(EPOCH FROM" in sql and "date_diff" not in sql
    else:
        assert sql == expected_sql


def test_a_dialect_the_compiler_does_not_know_is_refused(world: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="dialect"):
        compile(direct("refunds", "count"), world["graph"], "mysql")


# -- invalid IRs -------------------------------------------------------------------------------


def fields(ir: FeatureIR, graph: SchemaGraph) -> dict[str, str]:
    return {e.field: e.message for e in validate(ir, graph)}


def test_a_sum_over_a_text_column_is_rejected_at_the_column(world: dict[str, Any]) -> None:
    ir = direct("orders", "sum", column="status")
    errors = fields(ir, world["graph"])
    assert list(errors) == ["column"]
    assert "numeric" in errors["column"] and "status" in errors["column"]
    with pytest.raises(FeatureIRError) as raised:
        compile(ir, world["graph"])
    assert [e.field for e in raised.value.errors] == ["column"]


def test_a_path_that_is_not_in_the_graph_is_rejected_at_the_step(world: dict[str, Any]) -> None:
    ir = FeatureIR(
        name="bad_path",
        entity_table="customers",
        path=[edge("refunds", "products", "refund_id", "product_id")],
        source_table="refunds",
        agg="count",
    )
    errors = fields(ir, world["graph"])
    assert list(errors) == ["path[0]"] and "no such edge" in errors["path[0]"]
    with pytest.raises(FeatureIRError):
        compile(ir, world["graph"])


def test_a_path_must_be_connected_and_end_at_the_source(world: dict[str, Any]) -> None:
    disconnected = FeatureIR(
        name="jump",
        entity_table="customers",
        path=[ORDER_EDGE["order_items"]],  # starts at order_items/orders, not at customers
        source_table="order_items",
        agg="count",
    )
    assert "does not start at" in fields(disconnected, world["graph"])["path[0]"]
    wrong_end = direct("orders", "count").model_copy(update={"source_table": "refunds"})
    assert "ends at" in fields(wrong_end, world["graph"])["source_table"]


def test_the_first_edge_must_leave_the_entity_through_its_key(world: dict[str, Any]) -> None:
    ir = FeatureIR(
        name="from_products",
        entity_table="products",
        path=[PRODUCT_EDGE],
        source_table="order_items",
        agg="count",
    )
    assert fields(ir, world["graph"]) == {}  # product_id is the key of products: fine
    entity_orders = FeatureIR(
        name="orders_to_customers",
        entity_table="orders",
        path=[CUSTOMER_EDGE["orders"]],  # leaves orders through customer_id, not its key order_id
        source_table="customers",
        agg="count",
    )
    assert (
        "must leave the entity table through its key"
        in fields(entity_orders, world["graph"])["path[0]"]
    )


def test_a_table_without_an_event_time_must_be_confirmed_static(world: dict[str, Any]) -> None:
    graph = apply_overrides(world["graph"], SchemaOverrides(static_tables={"order_items": False}))
    errors = fields(via_orders("order_items", "count"), graph)
    assert list(errors) == ["path[1]"] and "not confirmed as static" in errors["path[1]"]


@pytest.mark.parametrize(
    ("ir", "field", "text"),
    [
        (direct("orders", "count", column="total"), "column", "does not take a column"),
        (direct("orders", "sum"), "column", "needs a column"),
        (direct("orders", "sum", column="nope"), "column", "no column"),
        (direct("orders", "count_distinct"), "column", "needs a column"),
        (direct("orders", "share"), "filter", "at least one"),
        (direct("orders", "count", window_days=0), "window_days", "between 1 and"),
        (direct("orders", "count", window_days=100000), "window_days", "between 1 and"),
        (direct("orders", "count", name="Bad Name"), "name", "lower-case"),
        (direct("orders", "sum", column="total", transform="log1p"), "transform", "log1p"),
        (
            direct("orders", "count", filter=[Predicate(column="nope", op="=", value=1)]),
            "filter[0].column",
            "no column",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="status", op="=", value=3)]),
            "filter[0].value",
            "text",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="total", op="=", value="x")]),
            "filter[0].value",
            "numeric",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="total", op="=", value=True)]),
            "filter[0].value",
            "numeric",
        ),
        (
            direct(
                "marketing_emails", "count", filter=[Predicate(column="opened", op=">", value=True)]
            ),
            "filter[0].op",
            "boolean",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="status", op="<", value="a")]),
            "filter[0].op",
            "text",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="status", op="in", value="a")]),
            "filter[0].value",
            "non-empty list",
        ),
        (
            direct("orders", "count", filter=[Predicate(column="status", op="is_null", value="a")]),
            "filter[0].value",
            "takes no value",
        ),
        (
            direct(
                "orders", "count", filter=[Predicate(column="ordered_at", op=">", value="soon")]
            ),
            "filter[0].value",
            "ISO date",
        ),
        (
            direct(
                "orders",
                "count",
                ratio_to=direct("sessions", "count").model_copy(update={"column": "pages"}),
            ),
            "ratio_to.column",
            "does not take a column",
        ),
        (
            direct(
                "orders",
                "count",
                ratio_to=direct("orders", "count", ratio_to=direct("orders", "count")),
            ),
            "ratio_to.ratio_to",
            "one level",
        ),
    ],
)
def test_each_kind_of_mistake_is_rejected_with_the_name_of_the_field(
    ir: FeatureIR, field: str, text: str, world: dict[str, Any]
) -> None:
    errors = fields(ir, world["graph"])
    assert field in errors, errors
    assert text in errors[field], errors[field]
    with pytest.raises(FeatureIRError):
        compile(ir, world["graph"])


def test_every_error_is_reported_at_once_not_just_the_first(
    world: dict[str, Any],
) -> None:
    ir = FeatureIR(
        name="Bad",
        entity_table="customers",
        path=[CUSTOMER_EDGE["orders"]],
        source_table="orders",
        agg="sum",
        column="status",
        window_days=0,
        filter=[Predicate(column="total", op="=", value="x")],
    )
    assert set(fields(ir, world["graph"])) == {"name", "column", "window_days", "filter[0].value"}


def test_shapes_pydantic_can_check_are_rejected_before_the_graph_is_consulted() -> None:
    base: dict[str, Any] = {
        "name": "x",
        "entity_table": "customers",
        "path": [CUSTOMER_EDGE["orders"]],
        "source_table": "orders",
    }
    with pytest.raises(ValidationError):
        FeatureIR(**base, agg="median")
    with pytest.raises(ValidationError):
        FeatureIR(**{**base, "path": []}, agg="count")
    with pytest.raises(ValidationError):
        FeatureIR(**base, agg="count", filter=[{"column": "total", "op": "like", "value": "x"}])
    with pytest.raises(ValidationError):
        FeatureIR(**base, agg="count", transform="sqrt")


def test_a_hostile_literal_stays_a_literal(world: dict[str, Any]) -> None:
    nasty = "x'; DROP TABLE orders; --"
    ir = direct("orders", "count", filter=[Predicate(column="status", op="=", value=nasty)])
    sql = compile(ir, world["graph"])
    assert "t1.status = 'x''; DROP TABLE orders; --'" in sql  # one quoted string, quote doubled
    result = run_feature(sql, world["tables"], world["labels"], world["graph"])
    assert (result["value"] == 0).all()


# -- property test -----------------------------------------------------------------------------

PATHS: dict[str, list[EdgeRef]] = {
    "orders": [CUSTOMER_EDGE["orders"]],
    "refunds": [CUSTOMER_EDGE["refunds"]],
    "sessions": [CUSTOMER_EDGE["sessions"]],
    "support_tickets": [CUSTOMER_EDGE["support_tickets"]],
    "marketing_emails": [CUSTOMER_EDGE["marketing_emails"]],
    "orders>order_items": [CUSTOMER_EDGE["orders"], ORDER_EDGE["order_items"]],
    "orders>refunds": [CUSTOMER_EDGE["orders"], ORDER_EDGE["refunds"]],
    "orders>order_items>products": [
        CUSTOMER_EDGE["orders"],
        ORDER_EDGE["order_items"],
        PRODUCT_EDGE,
    ],
}
WINDOWS = [None, 7, 30, 90, 365]


class Sampler:
    """Draws valid IRs, with filter values taken from the data so filters are not vacuous."""

    def __init__(self, world: dict[str, Any], rng: random.Random) -> None:
        self.graph: SchemaGraph = world["graph"]
        self.rng = rng
        self.tables = {t.key: t for t in self.graph.tables}
        self.frames = {name: data.to_pandas() for name, data in world["tables"].items()}

    def predicate(self, table: str) -> Predicate | None:
        columns = [
            c
            for c in self.tables[table].columns
            if type_kind(c.type) in ("integer", "float", "text", "boolean") and c.hint != "id"
        ]
        if not columns:
            return None
        column = self.rng.choice(columns)
        kind = type_kind(column.type)
        values = self.frames[table][column.name].dropna()
        if values.empty:
            return Predicate(column=column.name, op="is_null")
        if kind == "boolean":
            return Predicate(
                column=column.name,
                op=self.rng.choice(["=", "!="]),
                value=bool(self.rng.choice(values)),
            )
        if kind == "text":
            pool = sorted(values.unique())[:20]
            if self.rng.random() < 0.3:
                picks = self.rng.sample(pool, k=min(2, len(pool)))
                return Predicate(column=column.name, op="in", value=picks)
            return Predicate(
                column=column.name, op=self.rng.choice(["=", "!="]), value=self.rng.choice(pool)
            )
        number = float(values.quantile(self.rng.choice([0.25, 0.5, 0.75])))
        if kind == "integer":
            number = int(number)
        return Predicate(
            column=column.name, op=self.rng.choice(["<", "<=", ">", ">=", "="]), value=number
        )

    def ir(self, index: int, allow_ratio: bool = True) -> FeatureIR:
        key = self.rng.choice(sorted(PATHS))
        path = PATHS[key]
        source = key.split(">")[-1]
        table = self.tables[source]
        numeric = [c.name for c in table.columns if c.hint == "numeric"]
        aggs = ["count", "count_distinct", "days_since_last", "days_since_first", "share"]
        has_event = any(self.tables[t].time_column for t in key.split(">"))
        if not has_event:
            aggs = ["count", "count_distinct", "share"]
        if numeric:
            aggs += ["sum", "mean", "min", "max", "std"]
        agg = self.rng.choice(aggs)
        column = None
        if agg == "count_distinct":
            column = self.rng.choice(
                [c.name for c in table.columns if type_kind(c.type) != "timestamp"]
            )
        elif agg in ("sum", "mean", "min", "max", "std"):
            column = self.rng.choice(numeric)
        filters = [
            p for p in (self.predicate(source) for _ in range(self.rng.choice([0, 1, 2]))) if p
        ]
        if agg == "share" and not filters:
            filters = [p for p in [self.predicate(source)] if p] or [
                Predicate(column=table.primary_key[0], op="is_not_null")
            ]
        window = self.rng.choice(WINDOWS) if has_event else None
        transform = "none"
        if agg in ("count", "count_distinct", "std", "days_since_last", "days_since_first"):
            transform = self.rng.choice(["none", "none", "log1p"])
        ratio = None
        if allow_ratio and agg in ("count", "sum") and self.rng.random() < 0.3:
            ratio = self.ir(index, allow_ratio=False)
            ratio = ratio.model_copy(update={"transform": "none", "name": f"f{index}_den"})
        return FeatureIR(
            name=f"f{index}",
            entity_table="customers",
            path=path,
            source_table=source,
            agg=agg,  # type: ignore[arg-type]
            column=column,
            window_days=window,
            filter=filters,
            ratio_to=ratio,
            transform=transform,  # type: ignore[arg-type]
        )


def test_500_random_valid_irs_compile_pass_the_guard_and_ignore_the_future(
    world: dict[str, Any],
) -> None:
    sampler = Sampler(world, random.Random(100))
    graph: SchemaGraph = world["graph"]
    seen: set[str] = set()
    irs: list[FeatureIR] = []
    index = 0
    while len(irs) < 500:
        index += 1
        candidate = sampler.ir(index)
        key = candidate.model_dump_json(exclude={"name"})
        if key not in seen:
            seen.add(key)
            irs.append(candidate)
    checked = 0
    informative = 0
    aggs: dict[str, int] = {}
    for ir in irs:
        assert validate(ir, graph) == [], (ir, validate(ir, graph))
        sql = compile(
            ir, graph, "duckdb"
        )  # compile() runs the guard; run it again, on Postgres too
        for dialect in ("duckdb", "postgres"):
            verdict = check(compile(ir, graph, dialect), graph, dialect, allow_rewrite=False)
            assert verdict.status == "accepted", (ir, dialect, verdict.reasons)
        outcome = truncation_check(
            sql, world["tables"], graph, world["labels"], max_cutoffs=3, rows_per_cutoff=40
        )
        assert outcome.ok, (ir.model_dump_json(), outcome.mismatches[:3])
        checked += outcome.checked_rows
        values = run_feature(sql, world["tables"], world["labels"], graph)["value"].dropna()
        informative += values.nunique() >= 2
        aggs[ir.agg] = aggs.get(ir.agg, 0) + 1
    # the generator covers every aggregation, and most features are not constant
    assert set(aggs) == {
        "count",
        "count_distinct",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "days_since_last",
        "days_since_first",
        "share",
    }
    assert informative >= 350, informative
    print(
        f"\n500 random IRs: {aggs}; {checked} values compared with and without the future; "
        f"{informative} gave at least two distinct values"
    )
