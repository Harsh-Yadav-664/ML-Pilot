"""The prediction task spec (#49): valid specs on the demo database, field-level errors for bad ones.

The demo database (#47) is read into a schema graph once; every spec is validated against it with
the data ending on 2025-01-01. Each invalid spec below breaks one thing and must name the field.
"""

from __future__ import annotations

import copy
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks import spec as task_spec
from ml.tasks.conditions import ConditionError, parse_condition
from ml.tasks.spec import (
    SpecError,
    cutoff_dates,
    from_yaml,
    json_schema,
    parse_duration,
    schema_fingerprint,
    to_yaml,
    validate_against,
)
from tests.fixtures.demo_db import demo_sqlite

DATA_ENDS = datetime(2025, 1, 1, tzinfo=UTC)

CHURN = """
name: churn_30d
description: Customers who were active in the last 90 days and place no order in the next 30.
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
  - exists: {table: orders, where: "ordered_at >= :cutoff - interval '90 days' AND ordered_at < :cutoff"}
target:
  type: binary
  window: {start: ":cutoff", end: ":cutoff + interval '30 days'"}
  expression: {table: orders, agg: count, where: "status != 'cancelled'", compare: "= 0"}
horizon: 30d
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
metric: pr_auc
"""

# The demo database has no payments table, so "will not pay" is stood in for by a refund.
REFUND = """
name: refund_30d
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
  - exists: {table: orders, where: "ordered_at < :cutoff"}
target:
  type: binary
  expression: {table: refunds, agg: count, compare: "> 0"}
horizon: 30d
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 2 weeks}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
metric: roc_auc
"""

SPEND = """
name: spend_next_30d
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
target:
  type: regression
  expression: {table: orders, agg: sum, column: total, where: "status != 'cancelled'"}
horizon: 30 days
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
"""


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> SchemaGraph:
    path = demo_sqlite(Path(tmp_path_factory.mktemp("spec")) / "demo.sqlite")
    return build_schema_graph(open_source(ConnectionSpec(dialect="sqlite", database=str(path))))


def issues_of(graph: SchemaGraph, text: str) -> list[task_spec.SpecIssue]:
    return validate_against(from_yaml(text), graph, DATA_ENDS)


def errors_of(graph: SchemaGraph, text: str) -> dict[str, str]:
    return {i.path: i.message for i in issues_of(graph, text) if i.severity == "error"}


def changed(base: str, mutate) -> str:
    data = copy.deepcopy(yaml.safe_load(base))
    mutate(data)
    return yaml.safe_dump(data, sort_keys=False)


# -- valid specs ----------------------------------------------------------------------------


@pytest.mark.parametrize("text", [CHURN, REFUND, SPEND], ids=["churn", "refund", "spend"])
def test_the_demo_specs_parse_and_validate_without_errors(graph: SchemaGraph, text: str) -> None:
    parsed = from_yaml(text)
    assert [i for i in validate_against(parsed, graph, DATA_ENDS) if i.severity == "error"] == []
    assert from_yaml(to_yaml(parsed)) == parsed  # YAML round trip
    print(
        f"\n{parsed.name}: valid, {len(cutoff_dates(parsed))} cutoffs, metric {parsed.metric_name}"
    )


def test_a_regression_spec_is_valid_but_flagged_as_not_trainable_yet(graph: SchemaGraph) -> None:
    warnings = [i for i in issues_of(graph, SPEND) if i.severity == "warning"]
    assert [w.path for w in warnings] == ["target.type"]
    assert "cannot be trained yet" in warnings[0].message


def test_the_fixture_dates_give_the_cutoffs_the_docs_describe() -> None:
    parsed = from_yaml(CHURN)
    dates = cutoff_dates(parsed)
    assert dates[0] == date(2023, 4, 1) and dates[-1] == date(2024, 10, 1) and len(dates) == 19
    assert len(cutoff_dates(from_yaml(REFUND))) == 40  # every 2 weeks


def test_month_steps_keep_the_day_of_the_month_and_clamp_short_months() -> None:
    parsed = from_yaml(
        changed(
            CHURN,
            lambda d: d.update(
                cutoffs={"start": date(2024, 1, 31), "end": date(2024, 5, 31), "every": "1 month"}
            ),
        )
    )
    assert cutoff_dates(parsed) == [
        date(2024, 1, 31),
        date(2024, 2, 29),
        date(2024, 3, 31),
        date(2024, 4, 30),
        date(2024, 5, 31),
    ]


def test_default_metric_follows_the_task_type() -> None:
    assert from_yaml(changed(CHURN, lambda d: d.pop("metric"))).metric_name == "pr_auc"
    assert from_yaml(SPEND).metric_name == "mae"


def test_the_json_schema_lists_the_fields_for_the_editor() -> None:
    schema = json_schema()
    assert {"name", "entity", "target", "horizon", "cutoffs", "split"} <= set(schema["properties"])
    assert set(schema["required"]) == {"name", "entity", "target", "horizon", "cutoffs", "split"}


def test_the_schema_fingerprint_changes_with_the_schema(graph: SchemaGraph) -> None:
    other = graph.model_copy(deep=True)
    other.tables[0].columns.pop()
    assert schema_fingerprint(graph) == schema_fingerprint(graph.model_copy(deep=True))
    assert schema_fingerprint(other) != schema_fingerprint(graph)


# -- invalid specs: each names the field ----------------------------------------------------


def test_a_misspelt_and_a_missing_field_are_reported_at_their_names() -> None:
    text = changed(CHURN, lambda d: (d.pop("horizon"), d.update(horizn="30d")))
    with pytest.raises(SpecError) as e:
        from_yaml(text)
    got = {i.path: i.message for i in e.value.issues}
    assert got["horizn"] == "is not a known field" and got["horizon"] == "is required"


def test_a_nested_field_error_has_a_dotted_path() -> None:
    with pytest.raises(SpecError) as e:
        from_yaml(changed(CHURN, lambda d: d["cutoffs"].update(start="not a date")))
    assert [i.path for i in e.value.issues] == ["cutoffs.start"]


def test_text_that_is_not_yaml_or_not_a_mapping_is_refused_with_a_reason() -> None:
    with pytest.raises(SpecError, match="not valid YAML"):
        from_yaml("name: [unclosed")
    with pytest.raises(SpecError, match="mapping of fields"):
        from_yaml("- a\n- b\n")


def test_an_eligibility_rule_must_be_a_condition_or_exactly_one_exists_test() -> None:
    with pytest.raises(SpecError) as e:
        from_yaml(
            changed(
                CHURN,
                lambda d: d["eligibility"].append(
                    {"exists": {"table": "orders"}, "not_exists": {"table": "orders"}}
                ),
            )
        )
    assert "exactly one of exists or not_exists" in str(e.value)


def test_unknown_entity_table_and_key(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d["entity"].update(table="clients")))
    assert "no table 'clients'" in got["entity.table"]
    got = errors_of(graph, changed(CHURN, lambda d: d["entity"].update(key="cust_no")))
    assert "no column 'cust_no'" in got["entity.key"]
    got = errors_of(graph, changed(CHURN, lambda d: d["entity"].update(created_at="country")))
    assert "not a date or timestamp" in got["entity.created_at"]


def test_a_condition_with_a_function_a_subquery_or_a_stranger_placeholder_is_refused(
    graph: SchemaGraph,
) -> None:
    got = errors_of(
        graph, changed(CHURN, lambda d: d["eligibility"].__setitem__(0, "lower(country) = 'de'"))
    )
    assert "function LOWER is not allowed" in got["eligibility[0]"]
    got = errors_of(
        graph,
        changed(
            CHURN,
            lambda d: d["eligibility"].__setitem__(
                0, "customer_id IN (SELECT customer_id FROM orders)"
            ),
        ),
    )
    assert "subquery is not allowed" in got["eligibility[0]"]
    got = errors_of(
        graph, changed(CHURN, lambda d: d["eligibility"].__setitem__(0, "signup_at < :now"))
    )
    assert "unknown placeholder :now" in got["eligibility[0]"]


def test_a_condition_on_a_missing_column_names_the_column_and_table(graph: SchemaGraph) -> None:
    got = errors_of(
        graph, changed(CHURN, lambda d: d["eligibility"].__setitem__(0, "signed_up < :cutoff"))
    )
    assert "table 'customers' has no column 'signed_up'" in got["eligibility[0]"]


def test_an_exists_test_needs_a_foreign_key_to_the_entity(graph: SchemaGraph) -> None:
    got = errors_of(
        graph,
        changed(
            CHURN, lambda d: d["eligibility"].__setitem__(1, {"exists": {"table": "products"}})
        ),
    )
    assert "no foreign key to 'customers'" in got["eligibility[1].exists"]
    got = errors_of(
        graph,
        changed(
            CHURN,
            lambda d: d["eligibility"].__setitem__(
                1, {"exists": {"table": "orders", "where": "bogus = 1"}}
            ),
        ),
    )
    assert "no column 'bogus'" in got["eligibility[1].exists.where"]


def test_a_target_table_without_an_event_time_column_is_refused(graph: SchemaGraph) -> None:
    no_time = graph.model_copy(deep=True)
    next(t for t in no_time.tables if t.key == "orders").time_column = None
    got = {
        i.path: i.message
        for i in validate_against(from_yaml(CHURN), no_time, DATA_ENDS)
        if i.severity == "error"
    }
    assert "has no event-time column" in got["target.expression.table"]
    assert "has no event-time column" in got["eligibility[1].exists.table"]


def test_a_target_needs_a_column_for_sum_a_numeric_one_and_a_comparison_for_binary(
    graph: SchemaGraph,
) -> None:
    got = errors_of(graph, changed(SPEND, lambda d: d["target"]["expression"].pop("column")))
    assert got["target.expression.column"] == "sum needs a column"
    got = errors_of(
        graph, changed(SPEND, lambda d: d["target"]["expression"].update(column="status"))
    )
    assert "needs a numeric column" in got["target.expression.column"]
    got = errors_of(graph, changed(CHURN, lambda d: d["target"]["expression"].pop("compare")))
    assert "binary target needs the test" in got["target.expression.compare"]
    got = errors_of(
        graph, changed(CHURN, lambda d: d["target"]["expression"].update(compare="is zero"))
    )
    assert "not a comparison" in got["target.expression.compare"]
    got = errors_of(
        graph, changed(SPEND, lambda d: d["target"]["expression"].update(compare="> 0"))
    )
    assert "regression target is not compared" in got["target.expression.compare"]


def test_a_target_has_exactly_one_of_expression_and_expression_sql(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d["target"].update(expression_sql="SELECT 1")))
    assert "exactly one of expression or expression_sql" in got["target"]
    got = errors_of(graph, changed(CHURN, lambda d: d["target"].pop("expression")))
    assert "exactly one of" in got["target"]


def test_expression_sql_is_a_flagged_escape_hatch_that_must_be_a_select_using_cutoff(
    graph: SchemaGraph,
) -> None:
    def use_sql(sql: str):
        return lambda d: (d["target"].pop("expression"), d["target"].update(expression_sql=sql))

    ok = issues_of(
        graph,
        changed(CHURN, use_sql("SELECT customer_id, 1 FROM orders WHERE ordered_at > :cutoff")),
    )
    assert [i.severity for i in ok if i.path == "target.expression_sql"] == ["warning"]
    assert "flagged in the evidence report" in next(
        i.message for i in ok if i.path == "target.expression_sql"
    )
    got = errors_of(graph, changed(CHURN, use_sql("DELETE FROM orders")))
    assert got["target.expression_sql"] == "must be a single SELECT"
    got = errors_of(graph, changed(CHURN, use_sql("SELECT 1")))
    assert "must use :cutoff" in got["target.expression_sql"]


def test_horizon_and_every_must_be_durations(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d.update(horizon="soon")))
    assert "not a duration" in got["horizon"]
    got = errors_of(graph, changed(CHURN, lambda d: d.update(horizon="0d")))
    assert "longer than zero" in got["horizon"]
    got = errors_of(graph, changed(CHURN, lambda d: d["cutoffs"].update(every="monthly")))
    assert "not a duration" in got["cutoffs.every"]


def test_the_window_may_not_end_after_the_horizon_or_start_before_the_cutoff(
    graph: SchemaGraph,
) -> None:
    got = errors_of(
        graph,
        changed(CHURN, lambda d: d["target"]["window"].update(end=":cutoff + interval '60 days'")),
    )
    assert "past the horizon of 30 days" in got["target.window.end"]
    got = errors_of(
        graph,
        changed(CHURN, lambda d: d["target"]["window"].update(start=":cutoff - interval '7 days'")),
    )
    assert "target.window.start" in got
    got = errors_of(
        graph,
        changed(
            CHURN, lambda d: d["target"]["window"].update(start=":cutoff + interval '40 days'")
        ),
    )
    assert "must end after it starts" in got["target.window"]


def test_the_last_cutoff_plus_the_horizon_must_be_inside_the_data(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d["cutoffs"].update(end=date(2025, 1, 15))))
    assert "after the data ends (2025-01-01)" in got["cutoffs.end"]
    assert "Move cutoffs.end back to 2024-12-01" in got["cutoffs.end"]


def test_cutoffs_must_not_end_before_they_start(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d["cutoffs"].update(end=date(2023, 1, 1))))
    assert "is before cutoffs.start" in got["cutoffs.end"]


def test_split_dates_must_be_inside_the_cutoffs_in_order_and_a_horizon_apart(
    graph: SchemaGraph,
) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d["split"].update(val_from=date(2022, 1, 1))))
    assert "must lie inside the cutoff range" in got["split.val_from"]
    got = errors_of(graph, changed(CHURN, lambda d: d["split"].update(test_from=date(2025, 6, 1))))
    assert "must lie inside the cutoff range" in got["split.test_from"]
    got = errors_of(graph, changed(CHURN, lambda d: d["split"].update(test_from=date(2024, 3, 1))))
    assert "must be after split.val_from" in got["split.test_from"]
    got = errors_of(graph, changed(CHURN, lambda d: d["split"].update(test_from=date(2024, 4, 15))))
    assert "at least one horizon (30 days) after split.val_from" in got["split.test_from"]
    got = errors_of(graph, changed(CHURN, lambda d: d["split"].update(val_from=date(2023, 4, 10))))
    assert "at least one horizon (30 days) after cutoffs.start" in got["split.val_from"]


def test_every_part_needs_a_cutoff(graph: SchemaGraph) -> None:
    got = errors_of(
        graph,
        changed(CHURN, lambda d: d["cutoffs"].update(every="12 months")),
    )
    assert "no cutoff is left for test" in got["cutoffs"]


def test_metric_must_fit_the_task_type(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d.update(metric="mae")))
    assert "does not fit a binary task" in got["metric"]
    got = errors_of(graph, changed(SPEND, lambda d: d.update(metric="pr_auc")))
    assert "does not fit a regression task" in got["metric"]


def test_a_bad_name_is_refused(graph: SchemaGraph) -> None:
    got = errors_of(graph, changed(CHURN, lambda d: d.update(name="Churn 30d")))
    assert "lower-case letters" in got["name"]


def test_several_problems_are_reported_together_not_one_at_a_time(graph: SchemaGraph) -> None:
    def break_many(d):
        d["entity"]["table"] = "clients"
        d["horizon"] = "soon"
        d["metric"] = "mae"
        d["target"]["expression"].pop("compare")

    got = errors_of(graph, changed(CHURN, break_many))
    assert {"entity.table", "horizon", "metric", "target.expression.compare"} <= set(got)


def test_a_table_name_in_two_schemas_must_be_written_with_its_schema(graph: SchemaGraph) -> None:
    two = graph.model_copy(deep=True)
    orders = next(t for t in two.tables if t.key == "orders")
    orders.key, orders.db_schema = "public.orders", "public"
    twin = orders.model_copy(deep=True, update={"key": "archive.orders", "db_schema": "archive"})
    two.tables.append(twin)
    got = {
        i.path: i.message
        for i in validate_against(from_yaml(CHURN), two, DATA_ENDS)
        if i.severity == "error"
    }
    assert "is in several schemas" in got["eligibility[1].exists.table"]


# -- the condition language -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "status != 'cancelled'",
        "total > 10 AND plan IN ('a', 'b')",
        "ordered_at >= :cutoff - interval '90 days' AND ordered_at < :cutoff",
        "x IS NULL OR NOT (y LIKE 'a%')",
        "n BETWEEN 1 AND 5",
    ],
)
def test_conditions_that_the_language_allows(text: str) -> None:
    assert parse_condition(text).text == text


@pytest.mark.parametrize(
    "text, reason",
    [
        ("pg_sleep(10) IS NULL", "function"),
        ("CAST(x AS INT) > 1", "function CAST"),
        ("x > (SELECT 1)", "subquery"),
        ("x > 1; DROP TABLE orders", "single expression"),
        ("other.x > 1", "another table"),
        ("x > :user", "unknown placeholder"),
        ("d > :cutoff - interval 'a day'", "interval"),
        ("x >", "cannot parse"),
        ("", "empty"),
    ],
)
def test_conditions_that_the_language_refuses(text: str, reason: str) -> None:
    with pytest.raises(ConditionError, match=reason):
        parse_condition(text, table="orders")


def test_durations() -> None:
    assert str(parse_duration("30d")) == "30 days" and str(parse_duration("1 month")) == "1 month"
    assert parse_duration("2W").unit == "week"
    for bad in ("", "d", "-3d", "1.5d", "3 years"):
        with pytest.raises(ValueError):
            parse_duration(bad)
