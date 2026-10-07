"""The feature engine (#57): batch execution, the cache, sampling, time limit and budgets."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.features.baseline import top_values_from
from ml.features.dfs import generate
from ml.features.engine import (
    Budget,
    BudgetTracker,
    FeatureEngine,
    propose_within_budget,
    sample_entities,
)
from ml.tasks.pit_verify import FeatureTimeout
from tests.unit.test_dfs import demo, demo_baseline  # noqa: F401  (the demo world and its baseline)
from tests.unit.test_llm_sql import Scripted, answer, proposer

# ruff: noqa: F811


def engine_for(
    demo: dict[str, Any], cache: Path | None, version: str = "v1", **kw: Any
) -> FeatureEngine:
    return FeatureEngine(
        demo["graph"],
        demo["tables"],
        demo["labels"],
        data_version_id=version,
        label_version="labels-1",
        cache_dir=cache,
        **kw,
    )


def sql_of(demo: dict[str, Any], name: str) -> str:
    generated = generate(
        demo["graph"],
        demo["spec"].entity.table,
        entity_created_at=demo["spec"].entity.created_at,
        top_values=top_values_from(demo["tables"], demo["graph"]),
    )
    return next(c.sql for c in generated.candidates if c.name == name)


SLOW = (
    "SELECT l.entity_id, l.cutoff_time, count(*) AS value FROM __labels l, range(300000000) r "
    "GROUP BY l.entity_id, l.cutoff_time"
)


# -- the cache -------------------------------------------------------------------------------


def test_the_same_feature_is_computed_once_across_runs(demo, tmp_path: Path) -> None:
    sql = sql_of(demo, "orders__count_30d")
    cache = tmp_path / "features"
    with engine_for(demo, cache) as first:
        a = first.compute(sql)
        again = first.compute(sql)
    with engine_for(demo, cache) as second:  # a new run on the same data and labels
        b = second.compute(sql)
        reworded = second.compute(sql.replace("\n", " ").replace("SELECT", "select"))
    assert (a.status, again.status, b.status, reworded.status) == (
        "computed",
        "cached",
        "cached",
        "cached",
    )
    assert first.counts == {"computed": 1, "cached": 1, "failed": 0}
    assert second.counts == {"computed": 0, "cached": 2, "failed": 0}
    assert a.values is not None and b.values is not None
    pd.testing.assert_series_equal(a.values, b.values)
    assert len(a.values) == len(demo["labels"]) and a.values.notna().any()
    assert len(list(cache.glob("*.parquet"))) == 1
    print(
        f"\ncache: run 1 {first.counts}, run 2 {second.counts}; "
        f"{len(a.values)} values, computed in {a.seconds:.2f}s, read back in {b.seconds:.3f}s"
    )


def test_another_data_version_or_label_version_does_not_hit_the_cache(demo, tmp_path: Path) -> None:
    sql = sql_of(demo, "orders__count_30d")
    cache = tmp_path / "features"
    with engine_for(demo, cache, "v1") as one:
        assert one.compute(sql).status == "computed"
    with engine_for(demo, cache, "v2") as two:
        assert two.compute(sql).status == "computed"
    other_labels = FeatureEngine(
        demo["graph"],
        demo["tables"],
        demo["labels"],
        data_version_id="v1",
        label_version="labels-2",
        cache_dir=cache,
    )
    assert other_labels.compute(sql).status == "computed"
    other_labels.close()
    assert len(list(cache.glob("*.parquet"))) == 3


def test_a_failure_is_not_cached(demo, tmp_path: Path) -> None:
    with engine_for(demo, tmp_path / "c") as engine:
        bad = engine.compute("SELECT l.entity_id, l.cutoff_time FROM __labels l")
        assert bad.status == "failed" and bad.reason and bad.reason.startswith("contract:")
        assert not list((tmp_path / "c").glob("*.parquet"))


# -- the time limit ------------------------------------------------------------------------------


def test_a_query_over_its_time_limit_fails_with_timeout_and_the_next_one_runs(demo) -> None:
    good = sql_of(demo, "orders__count_30d")
    with engine_for(demo, None, timeout_s=0.5) as engine:
        slow = engine.compute(SLOW)
        after = engine.compute(good)
    assert slow.status == "failed" and slow.values is None
    assert slow.reason is not None and slow.reason.startswith("timeout:")
    assert slow.seconds < 10  # interrupted, not run to the end
    assert after.status == "computed" and after.values is not None
    assert engine.counts == {"computed": 1, "cached": 0, "failed": 1}
    print(
        f"\ntimeout: {slow.reason} (stopped after {slow.seconds:.2f}s); the next feature: {after.status}"
    )


# -- sampling ------------------------------------------------------------------------------------


def labels_frame(entities: int = 200, cutoffs: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for e in range(entities):
        positive = e % 10 == 0  # 10 percent of entities have positive rows
        for c in range(cutoffs):
            rows.append(
                (
                    f"e{e}",
                    pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * c),
                    int(positive and rng.random() < 0.8),
                )
            )
    return pd.DataFrame(rows, columns=["entity_id", "cutoff_time", "label"])


def test_below_the_limit_nothing_is_sampled() -> None:
    frame = labels_frame()
    out, info = sample_entities(frame, max_rows=10_000)
    assert out is frame and not info.sampled and info.fraction == 1.0 and info.rows_kept == 1000


def test_a_large_task_keeps_whole_entities_in_proportion_and_is_repeatable() -> None:
    frame = labels_frame()
    out, info = sample_entities(frame, max_rows=400, seed=7)
    again, _ = sample_entities(frame, max_rows=400, seed=7)
    other, _ = sample_entities(frame, max_rows=400, seed=8)
    assert info.sampled and info.rows_total == 1000 and info.rows_kept == len(out)
    assert info.fraction == pytest.approx(0.4)
    assert 395 <= len(out) <= 405
    # an entity is kept with all of its cutoffs, or not at all
    per_entity = out.groupby("entity_id").size()
    assert set(per_entity) == {5}
    # the entities that have a positive row are kept in the same proportion (20 of 200 -> 8)
    ever = frame.groupby("entity_id")["label"].max().gt(0)
    kept_positive = ever[per_entity.index].sum()
    assert kept_positive == round(ever.sum() * 0.4)
    pd.testing.assert_frame_equal(out, again)
    assert set(out["entity_id"]) != set(other["entity_id"])
    assert info.as_dict()["fraction"] == pytest.approx(0.4)


def test_the_engine_computes_on_the_sample_and_records_it(demo) -> None:
    total = len(demo["labels"])
    with engine_for(demo, None, max_rows=total // 2) as engine:
        assert engine.sampling.sampled and engine.sampling.rows_total == total
        assert len(engine.labels) < total
        done = engine.compute(sql_of(demo, "orders__count_30d"))
    assert done.values is not None and len(done.values) == len(engine.labels)


# -- budgets -------------------------------------------------------------------------------------


@dataclasses.dataclass
class Costly(Scripted):
    """A scripted model that reports a cost for every call (fake costs, for tests only)."""

    cost: float = 0.02

    async def complete_structured_result(self, task_type: Any, prompt: Any, schema: Any):  # type: ignore[no-untyped-def]
        out = await super().complete_structured_result(task_type, prompt, schema)
        return dataclasses.replace(out, cost_usd=self.cost)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, type_: str, **payload: Any) -> int:
        self.events.append((type_, payload))
        return len(self.events)


GOOD = ["good_refunds_14d", "good_cancelled_share", "good_ticket_count_60d"]


async def test_a_run_with_a_five_cent_budget_stops_and_says_so(demo, demo_baseline) -> None:
    gateway = Costly([answer(k) for k in GOOD])
    events = Recorder()
    tracker = BudgetTracker(Budget(max_cost_usd=0.05))
    run = await propose_within_budget(proposer(demo, demo_baseline, gateway), 10, tracker, events)
    assert run.stopped == "cost" and run.status == "stopped: budget (cost)"
    assert [r.status for r in run.records] == ["proposed"] * 3, [
        (r.status, r.stage, r.reasons) for r in run.records
    ]  # the third step finished
    assert run.used["cost_usd"] == pytest.approx(0.06) and run.used["proposals"] == 3
    assert len(gateway.prompts) == 3  # no fourth call
    kinds = [k for k, _ in events.events]
    assert kinds == ["feature_proposal"] * 3 + ["budget_stop"]
    assert events.events[-1][1]["budget"] == "cost"
    assert events.events[-1][1]["limits"]["max_cost_usd"] == 0.05
    print(f"\nbudget $0.05: {run.status}; used {run.used}")


async def test_the_proposal_and_time_budgets_stop_the_run_too(demo, demo_baseline) -> None:
    gateway = Costly([answer(k) for k in GOOD])
    tracker = BudgetTracker(Budget(max_proposals=2))
    run = await propose_within_budget(proposer(demo, demo_baseline, gateway), 10, tracker)
    assert run.stopped == "proposals" and len(run.records) == 2

    now = [0.0]
    clock = BudgetTracker(Budget(max_seconds=60), clock=lambda: now[0])
    assert clock.exceeded() is None
    now[0] = 61.0
    assert clock.exceeded() == "time"


async def test_a_run_that_finishes_inside_its_budget_is_not_stopped(demo, demo_baseline) -> None:
    gateway = Costly([answer(k) for k in GOOD[:2]])
    tracker = BudgetTracker(Budget(max_cost_usd=5.0))
    run = await propose_within_budget(proposer(demo, demo_baseline, gateway), 2, tracker)
    assert run.stopped is None and run.status == "completed" and len(run.records) == 2


async def test_a_timed_out_feature_is_a_rejected_proposal_and_the_run_continues(
    demo, demo_baseline, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = Costly(
        [
            answer("good_refunds_14d"),
            answer("good_cancelled_share"),
            answer("good_ticket_count_60d"),
        ]
    )
    p = proposer(demo, demo_baseline, gateway)
    real = p.engine._session.run
    calls = {"n": 0}

    def run(sql: str, *, timeout_s: float | None = None) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] <= 2:  # the proposal and its repair both run too long
            raise FeatureTimeout(f"the query ran longer than {timeout_s:g} seconds")
        return real(sql, timeout_s=timeout_s)

    monkeypatch.setattr(p.engine._session, "run", run)
    run_result = await propose_within_budget(p, 2, BudgetTracker(Budget()))
    first, second = run_result.records
    assert (first.status, first.stage) == ("rejected_guard", "execution")
    assert first.reasons[0].startswith("the query failed: timeout:")
    assert second.status == "proposed" and run_result.stopped is None
    assert p.engine.counts["failed"] == 2 and p.engine.counts["computed"] == 1
