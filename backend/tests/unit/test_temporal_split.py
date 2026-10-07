"""Temporal splits for relational tasks (#52): by cutoff date, with a gap so no label overlaps.

Every row is one (entity, cutoff) with the label window end (the label table's columns). The
checks are on the rows themselves: the largest label window end in training, and in validation,
is compared with the boundary it must not cross.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.schema_graph import build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.labels import run_on_source
from ml.tasks.spec import from_yaml
from ml.validation.splits import (
    RANDOM_SPLIT_REFUSED,
    SplitError,
    SplitPlan,
    TemporalSplitPlan,
    expanding_folds,
    make_splits,
    make_temporal_splits,
    partition_cutoffs,
    split_plan_for_task,
)
from ml.validation.strategies import TemporalCutoffStrategy
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_labels import CHURN

VAL_FROM = datetime(2024, 4, 1, tzinfo=UTC)
TEST_FROM = datetime(2024, 7, 1, tzinfo=UTC)
VAL_AT, TEST_AT = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")  # naive UTC, to compare
HORIZON = timedelta(days=30)


def labels(every_days: int = 7, entities: int = 20) -> pd.DataFrame:
    """Weekly cutoffs over two years, 30 day label windows, so windows cross the boundaries."""
    cutoffs = pd.date_range("2023-01-02", "2024-11-25", freq=f"{every_days}D")
    rows = [(e, c, c + HORIZON) for c in cutoffs for e in range(1, entities + 1)]
    return pd.DataFrame(rows, columns=["entity_id", "cutoff_time", "label_window_end"])


PLAN = TemporalSplitPlan(val_from=VAL_FROM, test_from=TEST_FROM, folds=3)


def test_no_training_window_passes_val_from_and_no_validation_window_passes_test_from() -> None:
    df = labels()
    s = make_temporal_splits(df.cutoff_time, df.label_window_end, PLAN)
    train_ends = df.label_window_end.iloc[s.train]
    val_ends = df.label_window_end.iloc[s.val]
    assert train_ends.max() <= VAL_AT
    assert val_ends.max() <= TEST_AT
    assert df.cutoff_time.iloc[s.val].min() >= VAL_AT
    assert df.cutoff_time.iloc[s.test].min() >= TEST_AT
    assert df.cutoff_time.iloc[s.train].max() < VAL_AT
    # the gap is real: weekly cutoffs inside the last horizon before each boundary are left out
    assert len(s.purged_train) > 0 and len(s.purged_val) > 0
    assert df.cutoff_time.iloc[s.purged_train].min() > VAL_AT - HORIZON
    print(
        f"\ntrain windows end by {train_ends.max():%Y-%m-%d} (val_from {VAL_AT:%Y-%m-%d}); "
        f"val windows end by {val_ends.max():%Y-%m-%d} (test_from {TEST_AT:%Y-%m-%d}); "
        f"purged {len(s.purged_train)} + {len(s.purged_val)} rows"
    )


def test_parts_are_disjoint_and_cover_every_row() -> None:
    df = labels()
    s = make_temporal_splits(df.cutoff_time, df.label_window_end, PLAN)
    parts = [s.train, s.val, s.test, s.purged_train, s.purged_val]
    flat = np.concatenate(parts)
    assert len(flat) == len(set(flat)) == len(df)


def test_a_window_ending_exactly_on_the_boundary_is_inside_and_one_second_later_is_purged() -> None:
    cut = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01",
                "2024-03-02",  # + 30 days = 2024-04-01 00:00:00: ends exactly at val_from
                "2024-03-02",
                "2024-04-10",
                "2024-08-01",
            ]
        )
    )
    end = pd.Series(
        pd.to_datetime(
            [
                "2024-01-31 00:00:00",
                "2024-04-01 00:00:00",
                "2024-04-01 00:00:01",
                "2024-07-01 00:00:00",
                "2024-08-31 00:00:00",
            ]
        )
    )
    parts = partition_cutoffs(list(zip(cut, end, strict=True)), PLAN)
    got = {k[1].isoformat(): v for k, v in parts.items()}
    assert got["2024-04-01T00:00:00"] == "train"
    assert got["2024-04-01T00:00:01"] == "purged_train"
    assert got["2024-07-01T00:00:00"] == "val"
    assert got["2024-08-31T00:00:00"] == "test"


def test_expanding_folds_train_on_the_past_only_and_never_touch_validation_or_test() -> None:
    df = labels()
    s = make_temporal_splits(df.cutoff_time, df.label_window_end, PLAN)
    assert len(s.folds) == 3
    train_rows = set(s.train)
    previous_size = 0
    for fold in s.folds:
        ft, fv = set(fold.train), set(fold.val)
        assert ft and fv and not ft & fv
        assert ft | fv <= train_rows  # nothing from validation, test or the purged gap
        assert df.cutoff_time.iloc[list(fold.val)].min() >= fold.val_start
        assert df.label_window_end.iloc[list(fold.train)].max() <= fold.val_start
        assert df.cutoff_time.iloc[list(fold.train)].max() < fold.val_start
        assert len(ft) > previous_size  # expanding window
        previous_size = len(ft)
    starts = [f.val_start for f in s.folds]
    assert starts == sorted(starts)


def test_fold_validation_blocks_do_not_overlap() -> None:
    df = labels()
    s = make_temporal_splits(df.cutoff_time, df.label_window_end, PLAN)
    seen: set[int] = set()
    for fold in s.folds:
        assert not seen & set(fold.val)
        seen |= set(fold.val)


def test_entities_in_several_parts_are_counted() -> None:
    df = labels(entities=20)
    s = make_temporal_splits(df.cutoff_time, df.label_window_end, PLAN, df.entity_id)
    assert s.entity_overlap == {
        "train_and_val": 20,
        "train_and_test": 20,
        "val_and_test": 20,
        "entities_in_train": 20,
        "entities_in_test": 20,
    }
    assert s.summary()["entity_overlap"]["train_and_test"] == 20


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (TemporalSplitPlan(val_from=TEST_FROM, test_from=VAL_FROM), "before test_from"),
        (
            TemporalSplitPlan(
                val_from=datetime(2022, 1, 1, tzinfo=UTC),
                test_from=datetime(2022, 6, 1, tzinfo=UTC),
            ),
            "No training rows",
        ),
        (
            TemporalSplitPlan(
                val_from=datetime(2024, 4, 1, tzinfo=UTC),
                test_from=datetime(2030, 1, 1, tzinfo=UTC),
            ),
            "No test rows",
        ),
    ],
)
def test_a_split_without_train_validation_or_test_is_an_error(
    plan: TemporalSplitPlan, message: str
) -> None:
    df = labels()
    with pytest.raises(SplitError, match=message):
        make_temporal_splits(df.cutoff_time, df.label_window_end, plan)


def test_too_few_training_cutoffs_for_the_folds_is_an_error() -> None:
    df = labels(every_days=60)
    with pytest.raises(SplitError, match="folds need at least"):
        make_temporal_splits(
            df.cutoff_time, df.label_window_end, PLAN.model_copy(update={"folds": 9})
        )


def test_a_window_that_ends_before_its_cutoff_is_an_error() -> None:
    with pytest.raises(SplitError, match="before its cutoff"):
        partition_cutoffs(
            [(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))], PLAN
        )


def test_timezone_aware_times_are_compared_as_utc() -> None:
    aware = pd.to_datetime(["2024-03-01T23:00:00-02:00"])  # = 2024-03-02 01:00 UTC
    parts = partition_cutoffs([(aware[0], aware[0] + HORIZON)], PLAN)
    ((cutoff, _),) = parts
    assert cutoff == pd.Timestamp("2024-03-02 01:00:00")


# -- a random split is refused for a relational task -----------------------------------------


def test_a_random_split_on_a_relational_task_is_refused_with_a_clear_message() -> None:
    y = pd.Series([0, 1] * 100)
    for plan in (SplitPlan(strategy="holdout"), SplitPlan(strategy="cv", inner_folds=3)):
        with pytest.raises(SplitError, match="cannot use a random split") as e:
            make_splits(y, plan, relational=True)
        assert str(e.value) == RANDOM_SPLIT_REFUSED
        print(f"\nrefused: {e.value}")
    spec = from_yaml(CHURN)
    for requested in ("holdout", "cv", "random"):
        with pytest.raises(SplitError, match="cannot use a random split"):
            split_plan_for_task(spec, requested)
    plan = split_plan_for_task(spec)
    assert (plan.val_from.date().isoformat(), plan.test_from.date().isoformat()) == (
        "2024-04-01",
        "2024-07-01",
    )
    assert split_plan_for_task(spec, "temporal") == plan


def test_the_ordinary_random_split_still_works_for_ordinary_tables() -> None:
    y = pd.Series([0, 1] * 200)
    s = make_splits(y, SplitPlan(strategy="holdout"))
    assert len(s.train) + len(s.val) + len(s.test) == 400


def test_the_temporal_plan_cannot_be_used_with_make_splits() -> None:
    with pytest.raises(SplitError, match="make_temporal_splits"):
        make_splits(pd.Series([0, 1] * 50), SplitPlan(strategy="temporal"))


# -- the validation strategy and the real labels ----------------------------------------------


def test_the_validation_strategy_returns_the_expanding_folds() -> None:
    df = labels()
    folds = TemporalCutoffStrategy(PLAN).split(df, "label")
    assert [f.fold for f in folds] == [0, 1, 2]
    assert all(f.train_indices and f.val_indices for f in folds)
    assert "folds=3" in TemporalCutoffStrategy(PLAN).describe()


def test_the_churn_labels_of_the_demo_database_split_without_overlap(tmp_path: Path) -> None:
    source = open_source(
        ConnectionSpec(dialect="duckdb", database=str(demo_duckdb(tmp_path / "demo.duckdb")))
    )
    graph = build_schema_graph(source)
    spec = from_yaml(CHURN)
    result = run_on_source(spec, graph, source, datetime(2025, 1, 1, tzinfo=UTC))
    frame = pd.DataFrame(source.query(result.compiled.sql, limit=10**7, timeout_s=300).to_pandas())
    plan = split_plan_for_task(spec)
    s = make_temporal_splits(frame.cutoff_time, frame.label_window_end, plan, frame.entity_id)
    end = pd.to_datetime(frame.label_window_end)
    assert end.iloc[s.train].max() <= pd.Timestamp(plan.val_from).tz_localize(None)
    assert end.iloc[s.val].max() <= pd.Timestamp(plan.test_from).tz_localize(None)
    summary = s.summary()
    print(
        f"\ndemo churn: train {summary['n_train']}, val {summary['n_val']}, test {summary['n_test']}, "
        f"purged {summary['n_purged_train']}+{summary['n_purged_val']}, "
        f"entities in train and test {summary['entity_overlap']['train_and_test']}"
    )
    assert summary["n_train"] > 1000 and summary["n_val"] > 500 and summary["n_test"] > 500


def test_expanding_folds_on_pairs_use_the_gap_rule() -> None:
    cutoffs = pd.date_range("2024-01-01", periods=12, freq="MS")
    pairs = [(c, c + HORIZON) for c in cutoffs]
    for train, val in expanding_folds(pairs, 3):
        start = min(c for c, _ in val)
        assert all(w <= start for _, w in train)


# -- bad input fails loudly (AGENTS.md rule 8) ----------------------------------------------


def test_a_missing_cutoff_or_window_end_is_an_error_not_a_silent_part() -> None:
    df = labels()
    bad_end = df.label_window_end.copy()
    bad_end.iloc[1] = pd.NaT
    with pytest.raises(SplitError, match="window_end has 1 missing"):
        make_temporal_splits(df.cutoff_time, bad_end, PLAN)
    bad_cut = df.cutoff_time.copy()
    bad_cut.iloc[0] = pd.NaT
    with pytest.raises(SplitError, match="cutoff_time has 1 missing"):
        make_temporal_splits(bad_cut, df.label_window_end, PLAN)
    with pytest.raises(SplitError, match="missing"):
        partition_cutoffs([(pd.Timestamp("2024-01-01"), pd.NaT)], PLAN)


@pytest.mark.filterwarnings("ignore:Could not infer format")
def test_times_that_cannot_be_read_are_a_split_error() -> None:
    df = labels()
    junk = df.cutoff_time.astype(object).copy()
    junk.iloc[0] = "not a date"
    with pytest.raises(SplitError, match="not a time"):
        make_temporal_splits(junk, df.label_window_end, PLAN)


def test_mixed_offsets_are_converted_to_utc_not_rejected() -> None:
    cut = ["2024-01-01T00:00:00+05:00", "2024-01-01T00:00:00+00:00", "2024-05-01 00:00:00"]
    end = ["2024-01-31T00:00:00+05:00", "2024-01-31T00:00:00+00:00", "2024-05-31 00:00:00"]
    parts = partition_cutoffs(list(zip(cut, end, strict=True)), PLAN)
    # +05:00 midnight is 19:00 the day before in UTC; all three keep their own part
    assert sorted(parts.values()) == ["train", "train", "val"]
