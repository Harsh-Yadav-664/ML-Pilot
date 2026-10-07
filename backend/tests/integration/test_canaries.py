"""Leakage canaries (#54): deliberate traps in the demo database, each shown to be caught.

A guarantee is only credible if it has been tested against traps. Each canary below states the
trap, the layer that must catch it and the expected catch; ``canaries`` runs all of them once on
the demo database (labels from the real churn task), the tests assert each one, and the last test
prints the "leakage canary report" that CI shows and the README quotes.

The layers: the point-in-time guard (``ml/tasks/pit_guard.py``, reads the SQL), its runtime check
(``pit_verify.truncation_check``, recomputes with the future deleted), the leakage checks on a
computed feature column (``ml/validation/leakage.py``) and the split refusal (#52).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ml.data.schema_graph import SchemaGraph, SchemaOverrides, apply_overrides, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.labels import run_on_source
from ml.tasks.pit_guard import check
from ml.tasks.pit_verify import run_feature, tables_from_source, truncation_check
from ml.tasks.spec import from_yaml
from ml.validation.leakage import scan
from ml.validation.splits import RANDOM_SPLIT_REFUSED, SplitError, SplitPlan, make_splits
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_labels import CHURN

SPEC = from_yaml(CHURN)
TARGET = SPEC.name  # the training table's label column carries the task name
TABLES = ["customers", "orders", "customer_status_snapshot"]

STATUS_FEATURE = """
SELECT l.entity_id, l.cutoff_time, MAX(CASE WHEN s.status = 'churned' THEN 1 ELSE 0 END) AS value
FROM __labels l JOIN customer_status_snapshot s ON s.customer_id = l.entity_id
GROUP BY l.entity_id, l.cutoff_time
"""
STATUS_FEATURE_BOUNDED = """
SELECT l.entity_id, l.cutoff_time, COUNT(*) AS value
FROM __labels l
JOIN customer_status_snapshot s ON s.customer_id = l.entity_id AND s.updated_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
"""
LABEL_AS_FEATURE = """
SELECT l.entity_id, l.cutoff_time, COUNT(*) AS value
FROM __labels l
JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at > l.cutoff_time
  AND o.ordered_at <= l.cutoff_time + INTERVAL 30 DAY
GROUP BY l.entity_id, l.cutoff_time
"""
IS_CHURNED = """
SELECT l.entity_id, l.cutoff_time, CAST(c.is_churned AS INTEGER) AS value
FROM __labels l JOIN customers c ON c.customer_id = l.entity_id
"""
DISCOUNT_AFTER_CHURN = """
SELECT l.entity_id, l.cutoff_time,
       CASE WHEN c.discount_code_used_after_churn IS NOT NULL THEN 1 ELSE 0 END AS value
FROM __labels l JOIN customers c ON c.customer_id = l.entity_id
"""
LAST_ORDER_NO_FILTER = """
SELECT l.entity_id, l.cutoff_time, MAX(o.ordered_at) AS value
FROM __labels l JOIN orders o ON o.customer_id = l.entity_id
GROUP BY l.entity_id, l.cutoff_time
"""


@dataclass
class Canary:
    id: int
    trap: str
    layer: str
    caught: bool
    evidence: str


class World:
    def __init__(self, path: Path) -> None:
        source = open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
        self.graph: SchemaGraph = build_schema_graph(source)
        self.tables = tables_from_source(source, TABLES)
        built = run_on_source(SPEC, self.graph, source, datetime(2025, 1, 1, tzinfo=UTC))
        frame = pd.DataFrame(
            source.query(built.compiled.sql, limit=10**7, timeout_s=300).to_pandas()
        )
        self.labels = frame.sample(2000, random_state=1)[["entity_id", "cutoff_time", "label"]]

    def feature_table(self, sql: str, name: str) -> pd.DataFrame:
        """The training table a feature would produce: its column named ``name`` and the label."""
        keys = self.labels[["entity_id", "cutoff_time"]]
        values = run_feature(sql, self.tables, keys, self.graph)
        merged = keys.merge(values, on=["entity_id", "cutoff_time"], how="left")
        return pd.DataFrame(
            {name: merged["value"].fillna(0).to_numpy(), TARGET: self.labels.label.to_numpy()}
        )

    def auc(self, table: pd.DataFrame, name: str) -> float:
        return float(roc_auc_score(table[TARGET], table[name]))


def reasons(sql: str, graph: SchemaGraph) -> list[str]:
    return [r.code for r in check(sql, graph).reasons]


def canary_1(w: World) -> Canary:
    rejected = [reasons(STATUS_FEATURE, w.graph), reasons(STATUS_FEATURE_BOUNDED, w.graph)]
    as_written = w.feature_table(STATUS_FEATURE, "customer_status")
    caught = all(r == ["last_modified_time"] for r in rejected)
    return Canary(
        1,
        "feature from customer_status_snapshot (rewritten after churn; only time column is "
        "updated_at), with and without a time bound",
        "PIT guard",
        caught,
        f"rejected {rejected[0][0]} both ways; as written the 'churned' flag has AUC "
        f"{w.auc(as_written, 'customer_status'):.3f}",
    )


def canary_2(w: World) -> Canary:
    guard = reasons(LABEL_AS_FEATURE, w.graph)
    table = w.feature_table(LABEL_AS_FEATURE, "orders_next_30d")
    found = [f for f in scan(table, TARGET) if f.check == "single_feature_predictiveness"]
    score = found[0].evidence["score"] if found else None
    caught = guard == ["reads_future"] and bool(found) and found[0].severity == "block"
    return Canary(
        2,
        "feature that is the label itself (orders in the 30 days after the cutoff)",
        "PIT guard, then single-feature check",
        caught,
        f"guard: {guard[0]}; if it got through, single-feature check blocks it (score {score})",
    )


def canary_3(w: World) -> Canary:
    static = apply_overrides(
        w.graph, SchemaOverrides(static_tables={"customer_status_snapshot": True})
    )
    result = check(STATUS_FEATURE, static)
    warned = any("static assumption on a table that changes" in x for x in result.warnings)
    return Canary(
        3,
        "customer_status_snapshot marked static by the user to get past canary 1",
        "evidence-report warning",
        result.status == "accepted" and warned,
        f"{result.status}, with the warning: {result.warnings[0] if result.warnings else 'none'}",
    )


def canary_4(w: World) -> Canary:
    guard = check(IS_CHURNED, w.graph).status  # a column of the entity table: bound by signup_at
    table = w.feature_table(IS_CHURNED, "is_churned")
    found = {f.check: f for f in scan(table, TARGET) if f.column == "is_churned"}
    name = found.get("name_tokens")
    return Canary(
        4,
        "customers.is_churned (set at the end of the simulation) used as a feature",
        "name-token check (SQL guard cannot see it)",
        name is not None and name.flagged,
        f"guard: {guard} (the row exists before the cutoff, the column is filled in later); "
        f"name check: {name.severity if name else 'missed'}; single-feature AUC "
        f"{w.auc(table, 'is_churned'):.3f} is below its warning limit",
    )


def canary_5(w: World) -> Canary:
    result = check(LAST_ORDER_NO_FILTER, w.graph)
    original = truncation_check(LAST_ORDER_NO_FILTER, w.tables, w.graph, w.labels)
    assert result.sql is not None
    fixed = truncation_check(result.sql, w.tables, w.graph, w.labels)
    return Canary(
        5,
        "MAX(ordered_at) with no cutoff filter",
        "PIT guard rewrite + runtime check",
        result.status == "rewritten" and not original.ok and fixed.ok,
        f"{result.status} ({result.rewrites[0]}); the original differs without the future in "
        f"{len(original.mismatches)} rows, the rewritten one in {len(fixed.mismatches)}",
    )


def canary_6(w: World) -> Canary:
    with pytest.raises(SplitError) as info:
        make_splits(w.labels.label, SplitPlan(strategy="holdout"), relational=True)
    return Canary(
        6,
        "random (holdout) split requested on the relational task",
        "split refusal (#52)",
        str(info.value) == RANDOM_SPLIT_REFUSED,
        "refused: " + str(info.value).split(". ")[0],
    )


def canary_7(w: World) -> Canary:
    table = w.feature_table(DISCOUNT_AFTER_CHURN, "discount_code_used_after_churn")
    found = [
        f
        for f in scan(table, TARGET)
        if f.column == "discount_code_used_after_churn" and f.check == "name_tokens"
    ]
    return Canary(
        7,
        "customers.discount_code_used_after_churn (only churned customers have one)",
        "name-token check",
        bool(found) and found[0].flagged,
        f"name check: {found[0].severity if found else 'missed'}; single-feature AUC "
        f"{w.auc(table, 'discount_code_used_after_churn'):.3f}",
    )


@pytest.fixture(scope="module")
def canaries(tmp_path_factory: pytest.TempPathFactory) -> list[Canary]:
    world = World(demo_duckdb(tmp_path_factory.mktemp("canary") / "demo.duckdb"))
    return [
        c(world) for c in (canary_1, canary_2, canary_3, canary_4, canary_5, canary_6, canary_7)
    ]


@pytest.mark.parametrize("number", range(1, 8))
def test_each_canary_is_caught(canaries: list[Canary], number: int) -> None:
    canary = canaries[number - 1]
    assert canary.caught, f"canary {canary.id} was not caught: {canary.evidence}"


def render(canaries: list[Canary]) -> str:
    width = max(len(c.layer) for c in canaries)
    lines = ["", "leakage canary report", "=" * 21]
    for c in canaries:
        lines.append(f"{c.id}. [{'CAUGHT' if c.caught else 'MISSED'}] {c.trap}")
        lines.append(f"   layer: {c.layer:<{width}}")
        lines.append(f"   {c.evidence}")
    caught = sum(c.caught for c in canaries)
    lines.append(f"{caught} of {len(canaries)} canaries caught")
    return "\n".join(lines)


def test_zz_the_canary_report(canaries: list[Canary]) -> None:
    text = render(canaries)
    print(text)
    assert all(c.caught for c in canaries), text
    assert f"{len(canaries)} of {len(canaries)} canaries caught" in text
