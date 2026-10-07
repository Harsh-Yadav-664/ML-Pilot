"""The context builder (#48): what each privacy level lets into a prompt, and what never goes in."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from ai.context_builder import (
    BuiltPrompt,
    ColumnContext,
    ContextBuilder,
    DatasetContext,
    PrivacyLevel,
    PrivacyPolicy,
    Relationship,
    TableContext,
)
from ai.context_inputs import dataset_from_profile
from ml.core.interfaces import ProfileResult

LABELS = [("organic_search", 480), ("paid_social", 320)]
SAMPLES = ["jane.doe@example.com", "john.roe@example.com"]


def dataset() -> DatasetContext:
    customers = TableContext(
        name="customers",
        row_count=10_000,
        time_column="signup_at",
        columns=[
            ColumnContext(
                "customer_id", "bigint", "id", primary_key=True, stats={"distinct": 10000}
            ),
            ColumnContext("email", "text", "text", stats={"distinct": 9990}, sample_values=SAMPLES),
            ColumnContext(
                "channel",
                "text",
                "categorical",
                stats={"distinct": 2, "null_fraction": 0.02},
                category_labels=LABELS,
                sample_values=["organic_search"],
            ),
            ColumnContext(
                "lifetime_value",
                "double precision",
                "numeric",
                stats={"mean": 412.37, "min": 0.0, "max": 9120.5, "p50": 250.0, "top": "SECRET"},
            ),
            ColumnContext("signup_at", "timestamp", "time", stats={"time_min": "2023-01-02"}),
        ],
    )
    orders = TableContext(
        name="orders",
        row_count=150_000,
        columns=[
            ColumnContext("order_id", "bigint", "id", primary_key=True),
            ColumnContext("customer_id", "bigint", "id"),
            ColumnContext("email", "text", "text"),
        ],
    )
    return DatasetContext(
        tables=[customers, orders],
        relationships=[Relationship("orders", ["customer_id"], "customers", ["customer_id"])],
    )


def build(level: PrivacyLevel, never_send: tuple[str, ...] = ()) -> BuiltPrompt:
    builder = ContextBuilder(PrivacyPolicy(level, frozenset(never_send)))
    return builder.prompt("test", "You propose features.").dataset(dataset()).build()


def test_schema_only_has_names_and_types_but_no_numbers_from_the_data():
    p = build(PrivacyLevel.schema_only)
    assert "Table customers" in p.text and "lifetime_value: double precision, numeric" in p.text
    assert "orders.customer_id -> customers.customer_id" in p.text
    for leaked in ("10,000", "412", "9120", "0.02", "2%", "organic_search", "jane.doe"):
        assert leaked not in p.text
    assert p.manifest.stats_used == []
    assert not p.manifest.category_labels_included and not p.manifest.sample_values_included


def test_schema_and_stats_adds_aggregates_but_no_labels_or_values():
    p = build(PrivacyLevel.schema_and_stats)
    assert "10,000 rows" in p.text and "mean 412.37" in p.text and "nulls 2.0%" in p.text
    assert "SECRET" not in p.text  # a statistic that is not on the list is dropped
    for leaked in ("organic_search", "paid_social", "jane.doe", "john.roe"):
        assert leaked not in p.text
    assert {"mean", "min", "max", "p50", "distinct", "null_fraction"} <= set(p.manifest.stats_used)
    assert not p.manifest.category_labels_included and not p.manifest.sample_values_included


def test_category_labels_are_opt_in():
    p = build(PrivacyLevel.allow_category_labels)
    assert "organic_search (480)" in p.text and "paid_social (320)" in p.text
    assert "jane.doe" not in p.text
    assert p.manifest.category_labels_included and not p.manifest.sample_values_included


def test_sample_values_are_opt_in_and_flagged():
    p = build(PrivacyLevel.allow_sample_values)
    assert "jane.doe@example.com" in p.text and "organic_search (480)" in p.text
    assert p.manifest.sample_values_included and p.manifest.category_labels_included


@pytest.mark.parametrize("level", list(PrivacyLevel))
def test_never_send_column_is_in_no_prompt_at_any_level(level):
    p = build(level, never_send=("email",))
    assert "email" not in (p.text + p.system + json.dumps(asdict(p.manifest))).lower()
    assert "jane.doe" not in p.text
    assert p.manifest.excluded_columns == 2  # customers.email and orders.email
    assert "customer_id" in p.text  # other columns stay


def test_table_qualified_never_send_hides_only_that_table():
    p = build(PrivacyLevel.schema_and_stats, never_send=("orders.email",))
    assert p.manifest.excluded_columns == 1
    assert p.manifest.tables["orders"] == ["order_id", "customer_id"]
    assert "email" in p.manifest.tables["customers"]


def test_a_relationship_through_an_excluded_column_is_dropped():
    p = build(PrivacyLevel.schema_and_stats, never_send=("customer_id",))
    assert "Relationships" not in p.text and "customer_id" not in p.text


def test_excluded_time_column_is_not_named_as_the_event_time():
    p = build(PrivacyLevel.schema_and_stats, never_send=("signup_at",))
    assert "signup_at" not in p.text and "event time" not in p.text


def test_never_send_names_are_replaced_in_free_text_facts_and_system():
    builder = ContextBuilder(PrivacyPolicy(never_send=frozenset({"Email", "customers.ssn"})))
    p = (
        builder.prompt("test", "Never mention email.")
        .text("Question", "Does the EMAIL domain predict churn? Try length(email) and ssn.")
        .facts(
            "History",
            [{"formula": "email_domain / ssn", "importances": {"email": 0.4, "age": 0.1}}],
        )
        .build()
    )
    blob = (p.system + p.text).lower()
    assert "email" not in blob and "ssn" not in blob
    assert "age" in blob and "[excluded column]" in p.text
    assert p.manifest.names_replaced >= 6


def test_unrelated_words_are_not_replaced():
    builder = ContextBuilder(PrivacyPolicy(never_send=frozenset({"age"})))
    p = builder.prompt("t").text("", "average usage and age").build()
    assert "average usage and [excluded column]" in p.text


def test_a_prompt_cannot_be_made_by_hand():
    with pytest.raises(TypeError, match="ContextBuilder"):
        BuiltPrompt("x", "", "raw text", ContextBuilder().prompt("x").build().manifest)


def test_policy_from_project_settings():
    assert PrivacyPolicy.from_settings({}) == PrivacyPolicy()
    policy = PrivacyPolicy.from_settings(
        {"privacy": {"level": "schema_only", "never_send": ["email", "orders.total"]}}
    )
    assert policy.level is PrivacyLevel.schema_only
    assert policy.excludes("orders", "total") and policy.excludes("any", "EMAIL")
    assert not policy.excludes("customers", "total")


def test_csv_profile_labels_stay_out_by_default():
    profile = ProfileResult(
        rows=100,
        columns=2,
        missing_rate=0.0,
        duplicate_rows=0,
        target_balance={},
        column_stats={
            "plan": {
                "dtype": "object",
                "count": 100,
                "missing_pct": 0.0,
                "n_unique": 2,
                "top_values": {"enterprise": 60, "starter": 40},
            },
            "age": {"dtype": "int64", "count": 100, "missing_pct": 0.0, "mean": 41.5, "std": 9.0},
        },
    )
    default = ContextBuilder().prompt("t").dataset(dataset_from_profile(profile)).build()
    assert "enterprise" not in default.text and "starter" not in default.text
    assert "mean 41.5" in default.text and "stddev 9" in default.text
    opted = (
        ContextBuilder(PrivacyPolicy(PrivacyLevel.allow_category_labels))
        .prompt("t")
        .dataset(dataset_from_profile(profile))
        .build()
    )
    assert "enterprise (60)" in opted.text
