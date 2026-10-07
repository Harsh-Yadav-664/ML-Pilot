"""No prompt carries a cell value of the demo database unless the project opted in (#48).

The prompt is built from the real schema graph and the real column statistics of the demo
database (including the top values the statistics hold), sent through the gateway with the
offline stub, and every distinct text and timestamp value of every table is searched for in what
the provider received.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ai.context_builder import ContextBuilder, PrivacyLevel, PrivacyPolicy
from ai.context_inputs import dataset_from_graph
from ai.gateway import AIGateway, PromptRecord
from ai.router import TaskType
from ml.data.engine import TableRef
from ml.data.profiling.db_stats import TableStats, profile_table
from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from tests.fixtures.demo_db import demo_sqlite
from tests.fixtures.gateway import StubOnlySettings

MIN_LENGTH = 5  # shorter strings ('US', 'pro') are words that a prompt may contain on its own


@pytest.fixture(scope="module")
def demo(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, SchemaGraph, dict[str, TableStats]]:
    path = demo_sqlite(tmp_path_factory.mktemp("demo") / "demo.sqlite")
    source = open_source(ConnectionSpec(dialect="sqlite", database=str(path)))
    graph = build_schema_graph(source)
    stats = {t.key: profile_table(source, TableRef(name=t.name)) for t in graph.tables}
    return path, graph, stats


def cell_values(path: Path) -> set[str]:
    """Every distinct text or timestamp value in the database (numbers are covered by the statistics)."""
    values: set[str] = set()
    with sqlite3.connect(path) as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            frame = pd.read_sql(f"SELECT * FROM {table}", con)
            for column in frame.columns:
                if not pd.api.types.is_numeric_dtype(frame[column]):
                    values |= {str(v) for v in frame[column].dropna().unique()}
    return {v for v in values if len(v) >= MIN_LENGTH and not v.replace(".", "").isdigit()}


async def sent_prompts(builder: ContextBuilder, graph: SchemaGraph, stats) -> list[PromptRecord]:
    gateway = AIGateway(config=StubOnlySettings())
    records: list[PromptRecord] = []
    gateway.add_prompt_hook(records.append)
    prompt = (
        builder.prompt("test.scan", "You propose features.")
        .dataset(dataset_from_graph(graph, stats))
        .text("Objective", "Which customers stop ordering in the next 30 days?")
        .build()
    )
    await gateway.complete(TaskType.HYPOTHESIZE, prompt)
    assert len(records) == 1
    return records


def found_in(records: list[PromptRecord], values: set[str]) -> set[str]:
    return {v for v in values if any(v in r.prompt or v in r.system for r in records)}


async def test_schema_and_stats_prompt_holds_no_cell_value(demo):
    path, graph, stats = demo
    schema_only = await sent_prompts(
        ContextBuilder(PrivacyPolicy(PrivacyLevel.schema_only)), graph, stats
    )
    # Words that are in the prompt anyway (column names, the objective) cannot be told from values.
    values = {v for v in cell_values(path) if not found_in(schema_only, {v})}
    # Time columns report their first and last timestamp, as numeric columns report min and max.
    allowed = {c.time_min or "" for t in stats.values() for c in t.columns} | {
        c.time_max or "" for t in stats.values() for c in t.columns
    }
    values -= allowed
    assert len(values) > 1000  # emails, discount codes, category labels, timestamps

    records = await sent_prompts(
        ContextBuilder(PrivacyPolicy(PrivacyLevel.schema_and_stats)), graph, stats
    )
    assert found_in(records, values) == set()


async def test_the_scan_does_find_labels_when_the_project_allows_them(demo):
    """The scan has teeth: at the opt-in level the same prompt does contain category labels."""
    _, graph, stats = demo
    labels = {
        t.value
        for table in stats.values()
        for c in table.columns
        for t in c.top_values
        if len(t.value) >= MIN_LENGTH
    }
    assert labels
    default = await sent_prompts(
        ContextBuilder(PrivacyPolicy(PrivacyLevel.schema_and_stats)), graph, stats
    )
    opted = await sent_prompts(
        ContextBuilder(PrivacyPolicy(PrivacyLevel.allow_category_labels)), graph, stats
    )
    schema_only = await sent_prompts(
        ContextBuilder(PrivacyPolicy(PrivacyLevel.schema_only)), graph, stats
    )
    baseline = found_in(schema_only, labels)
    assert found_in(opted, labels) - baseline  # labels appear
    assert found_in(default, labels) <= baseline  # and are absent by default


async def test_a_never_send_column_is_in_no_prompt(demo):
    path, graph, stats = demo
    policy = PrivacyPolicy(PrivacyLevel.allow_category_labels, frozenset({"email", "orders.total"}))
    records = await sent_prompts(ContextBuilder(policy), graph, stats)
    (record,) = records
    lines = record.prompt.splitlines()
    assert not [ln for ln in lines if ln.startswith(("  - email:", "  - total:"))]
    assert any(ln.startswith("  - status:") for ln in lines)  # the other orders columns stay
    # none of the removed columns' values either (they include the e-mail addresses)
    with sqlite3.connect(path) as con:
        emails = {r[0] for r in con.execute("SELECT email FROM customers LIMIT 500")}
    assert not any(e in record.prompt for e in emails)
