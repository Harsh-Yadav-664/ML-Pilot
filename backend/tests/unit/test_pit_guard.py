"""The point-in-time guard (#51): 70 queries with expected outcomes, the runtime check, and a
property test over 200 generated feature queries on the demo database.

Every query in tests/fixtures/pit_queries.yaml has an expected result (accepted, rewritten or
rejected, with the reason codes). Queries that are accepted or rewritten are also run: the
feature on the full data must equal the feature on data with every row at or after the cutoff
deleted. Queries marked ``leaky`` really read the future, and the runtime check must catch them.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ml.data.schema_graph import SchemaGraph, SchemaOverrides, apply_overrides, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.pit_guard import check
from ml.tasks.pit_verify import (
    ContractError,
    run_feature,
    tables_from_source,
    truncation_check,
)
from tests.fixtures.demo_db import demo_duckdb

CASES: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "fixtures" / "pit_queries.yaml").read_text()
)
TABLES = ["customers", "orders", "refunds", "sessions", "support_tickets", "marketing_emails"]
ALL_TABLES = [*TABLES, "order_items", "products"]
FIXED_CUTOFFS = ["2023-06-01", "2023-12-01", "2024-04-01"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    path = demo_duckdb(tmp_path_factory.mktemp("pit") / "demo.duckdb")
    source = open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
    graph = build_schema_graph(source)
    tables = tables_from_source(source, ALL_TABLES)
    orders = tables["orders"].to_pandas()
    customers = tables["customers"].to_pandas()
    rng = random.Random(3)
    entities = sorted(rng.sample(sorted(customers.customer_id.tolist()), 80))
    rows = [(e, pd.Timestamp(c)) for e in entities for c in FIXED_CUTOFFS]
    # one cutoff per entity placed exactly on one of its orders, so "<=" instead of "<" shows
    for e in entities:
        mine = orders[orders.customer_id == e].ordered_at
        if len(mine):
            rows.append((e, mine.sort_values().iloc[len(mine) // 2]))
    labels = pd.DataFrame(rows, columns=["entity_id", "cutoff_time"])
    return {"graph": graph, "tables": tables, "labels": labels}


def graph_for(world: dict[str, Any], static: list[str] | None) -> SchemaGraph:
    graph: SchemaGraph = world["graph"]
    if not static:
        return graph
    return apply_overrides(graph, SchemaOverrides(static_tables=dict.fromkeys(static, True)))


def ids(cases: list[dict[str, Any]]) -> list[str]:
    return [c["id"] for c in cases]


def test_the_fixture_has_enough_queries_of_every_kind() -> None:
    assert len(CASES) >= 25
    kinds = {c["expect"] for c in CASES}
    assert kinds == {"accepted", "rewritten", "rejected"}
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_the_expected_outcome(case: dict[str, Any], world: dict[str, Any]) -> None:
    result = check(case["sql"], graph_for(world, case.get("static")), case.get("dialect", "duckdb"))
    codes = [r.code for r in result.reasons]
    assert result.status == case["expect"], (result.status, result.reasons, result.sql)
    for code in case.get("codes", []):
        assert code in codes, (code, codes)
    if case["expect"] == "rejected":
        assert result.sql is None and result.reasons
    else:
        assert result.sql and not result.reasons
    if case["expect"] == "rewritten":
        assert result.rewrites
        assert case["contains"] in result.sql  # type: ignore[operator]
    else:
        assert not result.rewrites
    if case.get("static") and case["expect"] == "accepted":
        assert result.assumptions  # a static table is recorded, never silent


def runnable(case: dict[str, Any]) -> bool:
    return case["expect"] != "rejected" and case.get("runnable", True)


RUN = [c for c in CASES if runnable(c)]


@pytest.mark.parametrize("case", RUN, ids=ids(RUN))
def test_accepted_and_rewritten_features_do_not_change_when_the_future_is_deleted(
    case: dict[str, Any], world: dict[str, Any]
) -> None:
    graph = graph_for(world, case.get("static"))
    result = check(case["sql"], graph)
    assert result.sql is not None
    outcome = truncation_check(
        result.sql, world["tables"], graph, world["labels"], max_cutoffs=12, rows_per_cutoff=80
    )
    assert outcome.ok, outcome.mismatches[:3]
    assert outcome.checked_rows > 100


LEAKY = [c for c in CASES if c.get("leaky") and runnable(c)]


@pytest.mark.parametrize("case", LEAKY, ids=ids(LEAKY))
def test_the_runtime_check_catches_the_query_the_guard_had_to_rewrite(
    case: dict[str, Any], world: dict[str, Any]
) -> None:
    """The original query, run as written, does read the future; the check says so."""
    graph = graph_for(world, case.get("static"))
    outcome = truncation_check(
        case["sql"], world["tables"], graph, world["labels"], max_cutoffs=12, rows_per_cutoff=80
    )
    assert not outcome.ok
    first = outcome.mismatches[0]
    print(
        f"\n{case['id']}: leak at entity {first.entity_id}, {first.cutoff}: "
        f"{first.full} with the future, {first.truncated} without"
    )


def test_rows_following_frame_is_rejected_by_name(world: dict[str, Any]) -> None:
    sql = """
    SELECT l.entity_id, l.cutoff_time,
           sum(o.total) OVER (PARTITION BY l.entity_id, l.cutoff_time ORDER BY o.ordered_at
                              ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS value
    FROM __labels l JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time
    """
    result = check(sql, world["graph"])
    assert result.status == "rejected"
    assert [r.code for r in result.reasons] == ["following_frame"]
    print(
        f"\nROWS BETWEEN CURRENT ROW AND 1 FOLLOWING -> {result.status}: {result.reasons[0].message}"
    )


def test_a_date_typed_event_column_is_compared_in_whole_days(world: dict[str, Any]) -> None:
    """A date has no time of day, so ``d < cutoff`` would keep same-day events after the cutoff."""
    graph = world["graph"].model_copy(deep=True)
    for t in graph.tables:
        if t.name == "orders":
            for c in t.columns:
                if c.name == "ordered_at":
                    c.type = "DATE"
    plain = (
        "SELECT l.entity_id, l.cutoff_time, count(o.order_id) AS value FROM __labels l "
        "LEFT JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time "
        "GROUP BY l.entity_id, l.cutoff_time"
    )
    result = check(plain, graph)
    assert result.status == "rewritten" and "CAST(l.cutoff_time AS DATE)" in (result.sql or "")
    whole_days = plain.replace("< l.cutoff_time", "< CAST(l.cutoff_time AS DATE)")
    assert check(whole_days, graph).status == "accepted"


def test_the_contract_is_checked_at_run_time(world: dict[str, Any]) -> None:
    graph, tables, labels = world["graph"], world["tables"], world["labels"]
    duplicated = (
        "SELECT l.entity_id, l.cutoff_time, o.total AS value FROM __labels l "
        "JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time"
    )
    with pytest.raises(ContractError, match="more than one row"):
        run_feature(duplicated, tables, labels, graph)
    with pytest.raises(ContractError, match="not "):
        run_feature("SELECT entity_id, cutoff_time FROM __labels", tables, labels, graph)
    invented = (
        "SELECT customer_id AS entity_id, ordered_at AS cutoff_time, 1 AS value FROM orders "
        "WHERE customer_id = -1 UNION ALL SELECT -1, TIMESTAMP '2020-01-01', 1"
    )
    with pytest.raises(ContractError, match="not label rows"):
        run_feature(invented, tables, labels, graph)


# -- the property test: generated feature queries -------------------------------------------

CHILDREN: dict[str, dict[str, Any]] = {
    "orders": {
        "time": "ordered_at",
        "key": "order_id",
        "num": ["total"],
        "filters": ["status = 'completed'", "status = 'cancelled'", "total > 50"],
    },
    "refunds": {
        "time": "refunded_at",
        "key": "refund_id",
        "num": ["amount"],
        "filters": ["amount > 20"],
    },
    "sessions": {
        "time": "started_at",
        "key": "session_id",
        "num": ["pages"],
        "filters": ["device = 'mobile'", "pages > 3"],
    },
    "support_tickets": {
        "time": "opened_at",
        "key": "ticket_id",
        "num": [],
        "filters": ["category = 'billing'", "category = 'shipping'"],
    },
    "marketing_emails": {"time": "sent_at", "key": "email_id", "num": [], "filters": ["opened"]},
}
WINDOWS = [None, 7, 30, 90, 365]


def random_feature(rng: random.Random, with_bound: bool = True) -> str:
    """A feature in the shape the DFS templates produce: aggregate x table x window x filter."""
    table = rng.choice(sorted(CHILDREN))
    spec = CHILDREN[table]
    agg = rng.choice(
        ["count", "count_distinct", "sum", "avg", "min", "max"]
        if spec["num"]
        else ["count", "count_distinct"]
    )
    column = rng.choice(spec["num"]) if spec["num"] else spec["key"]
    expr = {
        "count": "count(t.{key})",
        "count_distinct": "count(DISTINCT t.{key})",
        "sum": "sum(t.{c})",
        "avg": "avg(t.{c})",
        "min": "min(t.{c})",
        "max": "max(t.{c})",
    }[agg].format(key=spec["key"], c=column)
    if agg in ("sum",):
        expr = f"coalesce({expr}, 0)"
    window = rng.choice(WINDOWS)
    flt = rng.choice([None, *spec["filters"]])
    time = f"t.{spec['time']}"
    conds = ["t.customer_id = l.entity_id"]
    if with_bound:
        conds.append(f"{time} < l.cutoff_time")
    if window:
        conds.append(f"{time} >= l.cutoff_time - INTERVAL '{window} days'")
    if flt:
        conds.append(f"t.{flt}")
    on = " AND ".join(conds)
    shape = rng.choice(["join", "subquery", "cte", "derived", "exists"])
    if shape == "join":
        return (
            f"SELECT l.entity_id, l.cutoff_time, {expr} AS value FROM __labels l "
            f"LEFT JOIN {table} t ON {on} GROUP BY l.entity_id, l.cutoff_time"
        )
    inner = f"SELECT {expr} FROM {table} t WHERE {on}"
    if shape == "subquery":
        return f"SELECT l.entity_id, l.cutoff_time, ({inner}) AS value FROM __labels l"
    if shape == "exists":
        return (
            f"SELECT l.entity_id, l.cutoff_time, EXISTS (SELECT 1 FROM {table} t WHERE {on}) "
            "AS value FROM __labels l"
        )
    grouped = (
        f"SELECT l.entity_id, l.cutoff_time, {expr} AS v FROM __labels l "
        f"JOIN {table} t ON {on} GROUP BY l.entity_id, l.cutoff_time"
    )
    if shape == "cte":
        return (
            f"WITH a AS ({grouped}) SELECT l.entity_id, l.cutoff_time, a.v AS value "
            "FROM __labels l LEFT JOIN a ON a.entity_id = l.entity_id AND a.cutoff_time = l.cutoff_time"
        )
    return (
        "SELECT l.entity_id, l.cutoff_time, d.v AS value FROM __labels l "
        f"LEFT JOIN ({grouped}) d ON d.entity_id = l.entity_id AND d.cutoff_time = l.cutoff_time"
    )


def test_200_generated_features_are_accepted_and_unchanged_by_deleting_the_future(
    world: dict[str, Any],
) -> None:
    rng = random.Random(51)
    queries = {random_feature(rng) for _ in range(400)}
    queries = set(sorted(queries)[:200]) if len(queries) >= 200 else queries
    assert len(queries) == 200, len(queries)
    graph = world["graph"]
    statuses: dict[str, int] = {}
    checked = 0
    for sql in sorted(queries):
        result = check(sql, graph)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        assert result.status == "accepted", (sql, result.reasons)
        assert result.sql is not None
        outcome = truncation_check(
            result.sql, world["tables"], graph, world["labels"], max_cutoffs=4, rows_per_cutoff=60
        )
        assert outcome.ok, (sql, outcome.mismatches[:3])
        checked += outcome.checked_rows
    print(
        f"\n200 generated queries: {statuses}; {checked} feature values compared with and without the future"
    )


def test_the_same_200_without_their_time_bound_are_rewritten_and_the_leak_is_detected(
    world: dict[str, Any],
) -> None:
    rng = random.Random(52)
    queries = sorted({random_feature(rng, with_bound=False) for _ in range(400)})[:200]
    graph = world["graph"]
    leaks = 0
    for sql in queries:
        result = check(sql, graph)
        assert result.status == "rewritten", (sql, result.status, result.reasons)
        assert result.sql is not None
        fixed = truncation_check(
            result.sql, world["tables"], graph, world["labels"], max_cutoffs=4, rows_per_cutoff=60
        )
        assert fixed.ok, (sql, fixed.mismatches[:3])
        raw = truncation_check(
            sql, world["tables"], graph, world["labels"], max_cutoffs=4, rows_per_cutoff=60
        )
        leaks += not raw.ok
    assert leaks >= 150  # a leak is found for most queries without their bound
    print(
        f"\nwithout the bound: {len(queries)} rewritten, the leak was visible in {leaks} as written"
    )


# -- last-modified times and explicit future reads (#54) -------------------------------------

STATUS = """
SELECT l.entity_id, l.cutoff_time, count(*) AS value
FROM __labels l JOIN customer_status_snapshot s
  ON s.customer_id = l.entity_id AND s.updated_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
"""


def test_a_last_modified_time_the_user_confirmed_as_the_event_time_is_allowed(
    world: dict[str, Any],
) -> None:
    graph = apply_overrides(
        world["graph"],
        SchemaOverrides(time_columns={"customer_status_snapshot": "updated_at"}),
    )
    table = next(t for t in graph.tables if t.name == "customer_status_snapshot")
    assert table.time_column_source == "user" and table.time_leakage_hint
    result = check(STATUS, graph)
    assert result.status == "accepted"  # their call, made in the schema
    assert len(result.warnings) == 1 and "you confirmed" in result.warnings[0]  # but not silent


def test_a_static_table_with_a_last_modified_time_warns_for_the_evidence_report(
    world: dict[str, Any],
) -> None:
    result = check(CASES[-1]["sql"], graph_for(world, ["customer_status_snapshot"]))
    assert result.status == "accepted" and result.assumptions
    # the static assumption, and (#139) the status column read from it
    assert len(result.warnings) == 2
    assert "static assumption on a table that changes" in result.warnings[0]
    assert "customer_status_snapshot.status may be overwritten" in result.warnings[1]
    assert result.as_dict()["warnings"] == result.warnings
    plain = check(CASES[0]["sql"], world["graph"])
    assert plain.warnings == []  # no warning when there is nothing to warn about


def test_the_future_is_refused_by_name_not_rewritten_into_an_empty_feature(
    world: dict[str, Any],
) -> None:
    result = check(
        """SELECT l.entity_id, l.cutoff_time, count(o.order_id) AS value
           FROM __labels l JOIN orders o ON o.customer_id = l.entity_id
             AND o.ordered_at > l.cutoff_time
           GROUP BY l.entity_id, l.cutoff_time""",
        world["graph"],
    )
    assert result.status == "rejected"
    assert [r.code for r in result.reasons] == ["reads_future"]
    assert "history" in result.reasons[0].message
