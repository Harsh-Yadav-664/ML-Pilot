"""Baseline features (#55): DFS candidates, the filters and cap, and the LightGBM baseline.

* On the demo database's churn task, every generated feature passes the point-in-time guard
  (for DuckDB and Postgres), a sample of them is unchanged when the future is deleted, and
  DFS features + LightGBM give a validation PR-AUC above the base rate, which is recorded.
* On a small synthetic schema, the cap, and the constant, near-constant and duplicate filters
  each do what they say, and the run is repeatable with the same seed.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from ml.data.engine import arrow_to_pandas
from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.features.baseline import BaselineError, BaselineResult, build_baseline, top_values_from
from ml.features.dfs import generate
from ml.tasks.labels import run_on_source
from ml.tasks.pit_guard import check
from ml.tasks.pit_verify import tables_from_source, truncation_check
from ml.tasks.spec import TaskSpec, from_yaml
from ml.validation.splits import TemporalSplitPlan
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_labels import CHURN

AS_OF = datetime(2025, 1, 1, tzinfo=UTC)


def world_for(path: Path, yaml_text: str) -> dict[str, Any]:
    source = open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
    graph = build_schema_graph(source)
    spec = from_yaml(yaml_text)
    run = run_on_source(spec, graph, source, AS_OF)
    labels = arrow_to_pandas(source.query(run.compiled.sql, limit=10**7, timeout_s=300))
    tables = tables_from_source(source, [t.name for t in graph.tables])
    return {"graph": graph, "spec": spec, "labels": labels, "tables": tables}


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    path = demo_duckdb(tmp_path_factory.mktemp("dfs") / "demo.duckdb")
    return world_for(path, CHURN)


@pytest.fixture(scope="module")
def demo_baseline(demo: dict[str, Any]) -> BaselineResult:
    plan = TemporalSplitPlan.from_spec(demo["spec"])
    return build_baseline(demo["spec"], demo["graph"], demo["tables"], demo["labels"], plan)


# -- the demo database -------------------------------------------------------------------------


def test_every_generated_feature_passes_the_guard_for_duckdb_and_postgres(
    demo: dict[str, Any],
) -> None:
    graph: SchemaGraph = demo["graph"]
    top = top_values_from(demo["tables"], graph)
    counts = {}
    for dialect in ("duckdb", "postgres"):
        generated = generate(
            graph,
            demo["spec"].entity.table,
            entity_created_at=demo["spec"].entity.created_at,
            top_values=top,
            dialect=dialect,
        )
        assert len(generated.candidates) > 100
        statuses = [
            check(c.sql, graph, dialect, allow_rewrite=False).status for c in generated.candidates
        ]
        assert statuses == ["accepted"] * len(statuses)  # 100%, not "most"
        counts[dialect] = len(statuses)
    assert counts["duckdb"] == counts["postgres"]
    print(
        f"\nguard: {counts['duckdb']} of {counts['duckdb']} DuckDB features and "
        f"{counts['postgres']} of {counts['postgres']} Postgres features accepted"
    )


def test_a_sample_of_the_features_ignores_the_future(demo: dict[str, Any]) -> None:
    graph: SchemaGraph = demo["graph"]
    generated = generate(
        graph,
        demo["spec"].entity.table,
        entity_created_at=demo["spec"].entity.created_at,
        top_values=top_values_from(demo["tables"], graph),
    )
    sample = random.Random(55).sample(generated.candidates, 25)
    labels = demo["labels"].copy()
    labels["cutoff_time"] = pd.to_datetime(labels["cutoff_time"], utc=True).dt.tz_localize(None)
    for cand in sample:
        outcome = truncation_check(
            cand.sql, demo["tables"], graph, labels, max_cutoffs=3, rows_per_cutoff=40
        )
        # the entity attributes are read from a table that is not rewritten after the cutoff
        assert outcome.ok, (cand.name, outcome.mismatches[:2])


def test_the_tables_that_cannot_be_placed_in_time_are_skipped_with_the_reason(
    demo: dict[str, Any],
) -> None:
    generated = generate(demo["graph"], "customers")
    reasons = {s.table: s.reason for s in generated.skipped}
    assert "last-modified" in reasons["customer_status_snapshot"]
    assert "no event time" in reasons["order_items"]
    assert not any("customer_status_snapshot" in c.name for c in generated.candidates)


def test_dfs_features_and_lightgbm_beat_the_base_rate_on_the_demo_churn_task(
    demo_baseline: BaselineResult,
) -> None:
    m = demo_baseline.metrics
    assert m["pr_auc"] > m["base_rate"] + 0.05, m
    assert m["pr_auc"] > m["trivial_pr_auc"]
    assert demo_baseline.engine == "lightgbm" and demo_baseline.seed == 42
    assert demo_baseline.split["n_test"] > 0 and "test" not in " ".join(m)  # test rows unscored
    top = demo_baseline.features[0]
    print(
        f"\nvalidation PR-AUC {m['pr_auc']:.4f} vs base rate {m['base_rate']:.4f} "
        f"({m['pr_auc'] / m['base_rate']:.2f}x), lift@10% {m['lift_at_10pct']:.2f}; "
        f"{len(demo_baseline.features)} features kept of {demo_baseline.candidates} candidates, "
        f"dropped {demo_baseline.dropped_counts()}; strongest: {top.candidate.name} "
        f"({top.importance:.1%} of gain); {demo_baseline.split['n_train']} train rows, "
        f"{demo_baseline.split['n_val']} validation rows"
    )


def test_a_post_outcome_column_of_the_entity_row_is_dropped_by_the_scan(
    demo_baseline: BaselineResult,
) -> None:
    gone = {d.name: d for d in demo_baseline.dropped}
    assert gone["customers__is_churned"].reason == "leakage_scan"
    kept = {f.candidate.name for f in demo_baseline.features}
    assert "customers__is_churned" not in kept
    assert {"customers__age_days", "customers__plan"} <= kept  # honest attributes stay


def test_every_kept_feature_carries_its_sql_and_a_sentence(demo_baseline: BaselineResult) -> None:
    for f in demo_baseline.features:
        c = f.candidate
        assert "__labels" in c.sql and "cutoff_time" in c.sql and c.description
        assert (c.ir is None) == (c.group == "attribute")
    assert abs(sum(f.importance for f in demo_baseline.features) - 1.0) < 1e-6


# -- a synthetic schema ------------------------------------------------------------------------

SYNTH_SPEC = """
name: quiet_30d
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
target:
  type: binary
  expression: {table: events, agg: count, where: "kind != 'noise'", compare: "= 0"}
horizon: 30d
cutoffs: {start: 2023-03-01, end: 2023-12-01, every: 1 month}
split: {val_from: 2023-09-01, test_from: 2023-11-01}
metric: pr_auc
"""


def synthetic_db(path: Path) -> Path:
    rng = random.Random(7)
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE customers (
          customer_id BIGINT PRIMARY KEY, signup_at TIMESTAMP, region VARCHAR, tier VARCHAR);
        CREATE TABLE events (
          event_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES customers(customer_id),
          happened_at TIMESTAMP, kind VARCHAR, amount DOUBLE);
        CREATE TABLE events_copy (
          event_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES customers(customer_id),
          happened_at TIMESTAMP, kind VARCHAR, amount DOUBLE);
        """
    )
    start = datetime(2022, 1, 1)  # noqa: DTZ001  (DuckDB TIMESTAMP is naive)
    event_id = 0
    for cid in range(1, 401):
        signup = start + timedelta(days=rng.randrange(0, 200))
        # 'north' for everybody (constant); 'b' for 2 customers of 400 (near constant)
        con.execute(
            "INSERT INTO customers VALUES (?, ?, 'north', ?)",
            [cid, signup, "b" if cid <= 2 else "a"],
        )
        rate = rng.choice([0.02, 0.06, 0.15])  # events per day: the signal for the target
        day = signup
        while day < datetime(2023, 12, 31):  # noqa: DTZ001
            day += timedelta(days=max(1, int(rng.expovariate(rate))))
            event_id += 1
            row = [
                event_id,
                cid,
                day + timedelta(hours=rng.randrange(24)),
                rng.choice(["buy", "view"]),
                7.0,
            ]
            con.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", row)
            con.execute("INSERT INTO events_copy VALUES (?, ?, ?, ?, ?)", row)
    con.close()
    return path


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return world_for(synthetic_db(tmp_path_factory.mktemp("synth") / "synth.duckdb"), SYNTH_SPEC)


def baseline(synth: dict[str, Any], **kw: Any) -> BaselineResult:
    spec: TaskSpec = synth["spec"]
    return build_baseline(
        spec,
        synth["graph"],
        synth["tables"],
        synth["labels"],
        TemporalSplitPlan.from_spec(spec),
        **kw,
    )


def test_a_copy_of_a_table_gives_duplicates_that_are_dropped(synth: dict[str, Any]) -> None:
    result = baseline(synth)
    dup = {d.name: d.detail for d in result.dropped if d.reason == "duplicate"}
    assert "events_copy__count_30d" in dup and dup["events_copy__count_30d"].startswith(
        "equal with"
    )
    kept = {f.candidate.name for f in result.features}
    assert "events__count_30d" in kept and not {n for n in kept if n.startswith("events_copy__")}
    assert len({d for d in dup if d.startswith("events_copy__")}) >= 10


def test_constant_and_near_constant_features_are_dropped(synth: dict[str, Any]) -> None:
    result = baseline(synth)
    dropped = {d.name: d for d in result.dropped}
    assert dropped["customers__region"].reason == "constant"  # 'north' for everybody
    assert dropped["customers__tier"].reason == "near_constant"  # 398 of 400 are 'a'
    assert "99" in dropped["customers__tier"].detail
    kept = {f.candidate.name for f in result.features}
    assert not kept & {"customers__region", "customers__tier"}


def test_the_cap_keeps_the_highest_priority_features(synth: dict[str, Any]) -> None:
    capped = baseline(synth, max_features=8)
    assert len(capped.features) == 8
    over = [d for d in capped.dropped if d.reason == "over_cap"]
    assert over and all(d.detail == "cap 8" for d in over)
    generated = generate(
        synth["graph"],
        "customers",
        entity_created_at="signup_at",
        top_values=top_values_from(synth["tables"], synth["graph"]),
    )
    priority = {c.name: c.priority for c in generated.candidates}
    # nothing that was cut for the cap ranks ahead of something that was kept
    assert min(priority[d.name] for d in over) >= max(
        priority[f.candidate.name] for f in capped.features
    )
    assert {f.candidate.group for f in capped.features} <= {"count", "recency", "trend"}
    full = baseline(synth)
    assert len(full.features) > 8
    assert not {d.name for d in capped.dropped if d.reason == "over_cap"} & {
        f.candidate.name for f in capped.features
    }


def test_the_same_seed_gives_the_same_features_and_scores(synth: dict[str, Any]) -> None:
    a, b = baseline(synth), baseline(synth)
    assert [f.candidate.name for f in a.features] == [f.candidate.name for f in b.features]
    assert a.metrics["pr_auc"] == b.metrics["pr_auc"]


def test_the_test_rows_never_reach_a_filter_or_the_model(synth: dict[str, Any]) -> None:
    result = baseline(synth)
    plan = TemporalSplitPlan.from_spec(synth["spec"])
    # shuffle the label of every test row: nothing the baseline reports may change
    labels = synth["labels"].copy()
    cutoff = pd.to_datetime(labels["cutoff_time"], utc=True).dt.tz_localize(None)
    test_rows = cutoff >= pd.Timestamp(plan.test_from).tz_localize(None)
    assert test_rows.sum() > 20
    labels.loc[test_rows, "label"] = 1 - labels.loc[test_rows, "label"]
    flipped = build_baseline(synth["spec"], synth["graph"], synth["tables"], labels, plan)
    assert flipped.metrics == result.metrics
    assert [f.candidate.name for f in flipped.features] == [
        f.candidate.name for f in result.features
    ]


def test_a_validation_set_with_one_class_fails_loudly(synth: dict[str, Any]) -> None:
    labels = synth["labels"].copy()
    cutoff = pd.to_datetime(labels["cutoff_time"], utc=True).dt.tz_localize(None)
    plan = TemporalSplitPlan.from_spec(synth["spec"])
    val = (cutoff >= pd.Timestamp(plan.val_from).tz_localize(None)) & (
        cutoff < pd.Timestamp(plan.test_from).tz_localize(None)
    )
    labels.loc[val, "label"] = 0
    with pytest.raises(BaselineError, match="validation rows have only one class"):
        build_baseline(synth["spec"], synth["graph"], synth["tables"], labels, plan)
