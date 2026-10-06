"""#93: ranking and calibration metrics against values computed by hand on 10 rows."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics.calibration import (
    ECE_LIMIT,
    brier_score,
    choose_threshold,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from ml.metrics.ranking import (
    fraction_key,
    k_for_fraction,
    lift_at_k,
    pr_auc,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
)

# Sorted by score, highest first. Positives at ranks 1, 3, 4 and 7; base rate 4/10.
Y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
P = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])


def test_precision_recall_and_lift_at_k():
    # Top 3 holds 2 positives: precision 2/3, recall 2/4, lift (2/3) / 0.4.
    assert precision_at_k(Y, P, 3) == pytest.approx(2 / 3)
    assert recall_at_k(Y, P, 3) == pytest.approx(0.5)
    assert lift_at_k(Y, P, 3) == pytest.approx(5 / 3)
    # Top 5: 3 positives.
    assert precision_at_k(Y, P, 5) == pytest.approx(0.6)
    assert lift_at_k(Y, P, 5) == pytest.approx(1.5)


def test_rows_are_ranked_by_score_not_by_position():
    shuffled = np.random.default_rng(0).permutation(10)
    assert precision_at_k(Y[shuffled], P[shuffled], 3) == pytest.approx(2 / 3)


def test_pr_auc_is_average_precision():
    # Precision at each positive's rank: 1/1, 2/3, 3/4, 4/7; their mean.
    assert pr_auc(Y, P) == pytest.approx((1 + 2 / 3 + 3 / 4 + 4 / 7) / 4)


def test_brier_score():
    # Sum of (p - y)^2 over the 10 rows is 1.925.
    assert brier_score(Y, P) == pytest.approx(0.1925)


def test_ece_and_reliability_table():
    # One row per bin, so ECE is the mean |p - y|: 3.6 / 10.
    assert expected_calibration_error(Y, P) == pytest.approx(0.36)
    table = reliability_table(Y, P)
    assert [row["count"] for row in table] == [1] * 10
    assert table[9] == {
        "bin": 9,
        "lower": 0.9,
        "upper": 1.0,
        "count": 1,
        "mean_predicted": 0.95,
        "observed_rate": 1.0,
    }


def test_ece_weights_bins_by_rows():
    # Bin [0.2, 0.3): p 0.2, 0.2 vs outcomes 1, 0 -> gap |0.2 - 0.5|, weight 2/4.
    # Bin [0.8, 0.9): p 0.8, 0.8 vs outcomes 1, 1 -> gap 0.2, weight 2/4.
    y, p = np.array([1, 0, 1, 1]), np.array([0.2, 0.2, 0.8, 0.8])
    assert expected_calibration_error(y, p) == pytest.approx(0.5 * 0.3 + 0.5 * 0.2)


def test_ranking_metrics_names_and_trivial_baseline():
    m = ranking_metrics(Y, P, fractions=(0.1, 0.5), absolute_k=(3, 50))
    assert m["base_rate"] == pytest.approx(0.4)
    assert m["trivial_pr_auc"] == pytest.approx(0.4)
    assert m["precision_at_10pct"] == 1.0  # top 1 row
    assert m["lift_at_10pct"] == pytest.approx(2.5)
    assert m["recall_at_50pct"] == pytest.approx(0.75)
    assert m["lift_at_3"] == pytest.approx(5 / 3)
    assert "lift_at_50" not in m  # more than the rows there are


def test_cutoff_helpers():
    assert k_for_fraction(7043, 0.10) == 705
    assert k_for_fraction(10, 0.10) == 1
    assert k_for_fraction(10, 0.01) == 1
    assert fraction_key(0.05) == "5pct" and fraction_key(0.005) == "0_5pct"


def test_threshold_maximises_f1_on_the_given_rows():
    # F1 by cut: top 1 = 0.4, top 3 = 4/7, top 4 = 0.75 (best), top 7 = 8/11.
    t = choose_threshold(Y, P)
    assert t["value"] == pytest.approx(0.65)
    assert t["val_f1"] == pytest.approx(0.75)
    assert t["objective"] == "max_f1"


def test_well_calibrated_probabilities_are_left_alone():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    calibrator, record = fit_calibrator(y, p)
    assert record["method"] == "none" and record["val_ece_before"] <= ECE_LIMIT
    assert np.array_equal(calibrator.apply(p), p)


@pytest.mark.parametrize(("rows", "method"), [(5000, "isotonic"), (400, "platt")])
def test_overconfident_probabilities_are_calibrated(rows, method):
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.2, 0.8, size=rows)
    y = (rng.uniform(size=rows) < true_p).astype(int)
    overconfident = np.clip((true_p - 0.5) * 2.5 + 0.5, 0, 1)
    calibrator, record = fit_calibrator(y, overconfident)
    assert record["method"] == method
    assert record["val_ece_before"] > ECE_LIMIT
    assert record["val_ece_after"] < record["val_ece_before"]
    # Monotone: calibration never reorders rows.
    order = np.argsort(overconfident)
    assert np.all(np.diff(calibrator.apply(overconfident)[order]) >= -1e-12)
