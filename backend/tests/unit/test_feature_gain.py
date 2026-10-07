"""The gain rule on time-ordered folds (ml/features/gain.py, #58), on a synthetic task."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from ml.features.gain import FoldScorer, compare
from ml.validation.splits import SplitError, TemporalSplitPlan, make_temporal_splits

START = pd.Timestamp("2022-01-01")
CUTOFFS = 30  # monthly


def world(seed: int = 0, entities: int = 400) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Label = 1 when a hidden score is high. ``signal`` carries the score, ``noise`` does not,
    ``weak`` carries it behind a lot of noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(CUTOFFS):
        cutoff = START + pd.DateOffset(months=c)
        hidden = rng.normal(size=entities)
        label = (hidden + rng.normal(scale=0.7, size=entities) > 0.8).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "cutoff": cutoff,
                    "end": cutoff + pd.Timedelta(days=30),
                    "label": label,
                    "base": rng.normal(size=entities),
                    "signal": hidden + rng.normal(scale=0.3, size=entities),
                    "weak": hidden * 0.05 + rng.normal(size=entities),
                    "noise": rng.normal(size=entities),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    plan = TemporalSplitPlan(
        val_from=datetime(2024, 1, 1, tzinfo=UTC),
        test_from=datetime(2024, 4, 1, tzinfo=UTC),
        folds=3,
    )
    split = make_temporal_splits(frame["cutoff"], frame["end"], plan)
    return frame, frame["label"].to_numpy(), split


def scores(frame: pd.DataFrame, y: np.ndarray, split: object, columns: list[str]) -> list[float]:
    scorer = FoldScorer(y, split.folds, train_rows=split.train)  # type: ignore[attr-defined]
    return scorer.score(frame[columns])


def test_a_feature_that_carries_the_signal_is_accepted_and_noise_is_not() -> None:
    frame, y, split = world()
    base = scores(frame, y, split, ["base"])
    with_signal = compare(base, scores(frame, y, split, ["base", "signal"]))
    with_noise = compare(base, scores(frame, y, split, ["base", "noise"]))
    assert with_signal.accepted and with_signal.mean_gain > 0.2
    assert with_signal.ci95[0] > 0  # the whole interval is above zero
    assert not with_noise.accepted and with_noise.mean_gain < with_noise.margin
    assert with_signal.metric == "pr_auc" and len(with_signal.base_scores) == 3
    assert with_signal.rule["validation"] == "temporal"
    print(
        f"\nsignal: gain {with_signal.mean_gain:+.3f} (margin {with_signal.margin:.3f}) accepted; "
        f"noise: gain {with_noise.mean_gain:+.3f} (margin {with_noise.margin:.3f}) rejected"
    )


def test_the_noise_feature_is_rejected_on_many_draws_of_the_world() -> None:
    accepted = 0
    for seed in range(1, 9):
        frame, y, split = world(seed, entities=200)
        base = scores(frame, y, split, ["base"])
        accepted += compare(base, scores(frame, y, split, ["base", "noise"])).accepted
    assert accepted <= 1, f"{accepted} of 8 noise features were accepted"
    print(f"\nnoise accepted in {accepted} of 8 synthetic worlds")


def test_the_rule_is_the_paired_one() -> None:
    result = compare(
        [0.50, 0.52, 0.51], [0.55, 0.53, 0.60], {"min_gain": 0.002, "std_multiplier": 1.0}
    )
    diffs = np.array([0.05, 0.01, 0.09])
    assert result.mean_gain == pytest.approx(diffs.mean())
    assert result.std_gain == pytest.approx(diffs.std(ddof=1))
    assert result.margin == pytest.approx(max(0.002, diffs.std(ddof=1)))
    assert result.accepted == (diffs.mean() > result.margin)
    assert compare([0.5, 0.5], [0.5, 0.5]).accepted is False  # no gain is not a gain
    with pytest.raises(ValueError):
        compare([0.5], [0.5, 0.6])


def test_the_folds_use_training_rows_only_and_a_foreign_row_is_refused() -> None:
    frame, y, split = world()
    test_rows = set(split.test.tolist())  # type: ignore[attr-defined]
    for fold in split.folds:  # type: ignore[attr-defined]
        assert not (set(fold.train.tolist()) | set(fold.val.tolist())) & test_rows
        assert (
            fold.train.max() < len(frame)
            and frame["cutoff"].iloc[fold.train].max() < frame["cutoff"].iloc[fold.val].min()
        )
    with pytest.raises(SplitError, match="not training rows"):
        FoldScorer(y, split.folds, train_rows=split.train[:10])  # type: ignore[attr-defined]


def test_the_same_inputs_give_the_same_scores() -> None:
    frame, y, split = world()
    assert scores(frame, y, split, ["base", "signal"]) == scores(
        frame, y, split, ["base", "signal"]
    )
