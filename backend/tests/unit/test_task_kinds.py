"""What differs between binary, regression and multi-class tasks (#104)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.experiments.acceptance import decide
from ml.features.gain import compare
from ml.metrics.kinds import UNSEEN, TaskKind


def test_a_lower_mae_is_a_gain_measured_as_a_relative_reduction() -> None:
    mae = TaskKind(task="regression")
    result = compare([10.0, 20.0, 40.0], [9.0, 18.0, 36.0], task=mae)  # 10% better on every fold
    assert result.accepted and result.mean_gain == pytest.approx(0.10)
    assert result.metric == "mae" and result.rule["gain"] == "relative reduction"
    worse = compare([10.0, 20.0, 40.0], [11.0, 22.0, 44.0], task=mae)
    assert not worse.accepted and worse.mean_gain < 0


def test_the_higher_is_better_rule_is_unchanged_for_binary_tasks() -> None:
    result = decide([0.50, 0.52, 0.51], [0.55, 0.56, 0.57], metric="pr_auc")
    assert result.accepted and result.mean_gain == pytest.approx(0.05, abs=0.01)
    assert "gain" not in result.rule


def test_spend_that_is_mostly_zero_is_fitted_with_tweedie_other_targets_are_not() -> None:
    zero_inflated = pd.Series([0.0] * 70 + [5.0] * 30)
    assert TaskKind.fit("regression", zero_inflated).objective == "tweedie"
    assert TaskKind.fit("regression", pd.Series([1.0, 2.0, 3.0, 4.0])).objective == "regression"
    assert TaskKind.fit("regression", pd.Series([-1.0] * 70 + [5.0] * 30)).objective == "regression"


def test_classes_come_from_the_training_rows_and_an_unseen_class_is_marked_not_dropped() -> None:
    kind = TaskKind.fit("multiclass", pd.Series(["b", "a", "b", "c"]))
    assert kind.classes == ("a", "b", "c")
    assert list(kind.encode(pd.Series(["c", "a", "never_seen"]))) == [2, 0, UNSEEN]


def test_multiclass_metrics_leave_out_unseen_rows_and_count_them() -> None:
    kind = TaskKind(task="multiclass", objective="multiclass", classes=("a", "b"))
    y = np.array([0, 1, 1, UNSEEN])
    pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])
    metrics = kind.evaluate(y, pred, reference=np.array([0, 1, 1, 1]))
    assert metrics["n_unseen_class"] == 1 and metrics["accuracy"] == 1.0
    assert metrics["log_loss"] < metrics["trivial_log_loss"]
    assert kind.headline(metrics) == {
        "metric": "log_loss",
        "value": metrics["log_loss"],
        "trivial": metrics["trivial_log_loss"],
    }


def test_regression_metrics_include_the_trivial_median_and_value_captured() -> None:
    kind = TaskKind(task="regression")
    y = np.array([0.0, 0.0, 10.0, 30.0])
    metrics = kind.evaluate(
        y, np.array([1.0, 0.0, 12.0, 25.0]), reference=np.array([0.0, 0.0, 4.0])
    )
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["trivial_mae"] == pytest.approx(10.0)  # always predicting the training median 0
    assert metrics["captured_value_at_10pct"] == pytest.approx(0.75)  # the top row holds 30 of 40
