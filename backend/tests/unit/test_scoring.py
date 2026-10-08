"""The pure parts of batch scoring (#62)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.scoring import score as s


def graph(**tables: list[str]) -> Any:
    return SimpleNamespace(
        tables=[
            SimpleNamespace(key=k, name=k, columns=[SimpleNamespace(name=c) for c in cols])
            for k, cols in tables.items()
        ]
    )


def test_a_missing_column_is_named_with_the_features_that_read_it() -> None:
    now = graph(orders=["id", "customer_id", "placed_at"])
    problems = s.required_columns_missing(
        {"orders": ["id", "total", "placed_at"], "gone": ["x"]},
        now,
        {"sum_total": "SELECT SUM(total) FROM orders", "other": "SELECT 1"},
    )
    assert problems == [
        "table gone is missing",
        "column orders.total is missing (used by sum_total)",
    ]


def test_nothing_is_reported_when_the_schema_still_has_everything() -> None:
    now = graph(orders=["id", "total", "extra"])
    assert s.required_columns_missing({"orders": ["id", "total"]}, now, {}) == []


def test_the_ranking_has_rank_decile_and_reasons_in_score_order() -> None:
    ids = pd.Series(list("abcdefghij") * 2)
    score = np.linspace(0.05, 0.95, 20)[::-1].copy()
    reasons = [[("f1", 0.5), ("f2", -0.25)] for _ in range(20)]
    rows = s.rank_rows(ids, score, reasons)
    assert list(rows["rank"]) == list(range(1, 21))
    assert rows["score"].is_monotonic_decreasing
    assert rows["decile"].iloc[0] == 1 and rows["decile"].iloc[-1] == 10
    assert rows["decile"].value_counts().eq(2).all()
    assert rows["reason_1"].iloc[0] == "f1 (+0.500)"
    assert rows["reason_2"].iloc[0] == "f2 (-0.250)"
    assert rows["reason_3"].iloc[0] == ""


def test_reasons_plus_the_base_value_add_up_to_the_raw_score() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (x["a"] + 0.2 * rng.normal(size=400) > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7},
        lgb.Dataset(x, y),
        num_boost_round=20,
    )
    score, reasons = s.predict_with_reasons(booster, x, k=2)
    assert np.allclose(score, booster.predict(x))
    assert all(len(r) == 2 for r in reasons)
    # the strongest reason of nearly every row is the feature that actually drives the label
    assert sum(r[0][0] == "a" for r in reasons) > 350


def stats(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return s.feature_stats(frame, np.arange(len(frame)))


def test_drift_is_warned_about_never_raised() -> None:
    train = pd.DataFrame({"x": np.arange(100, dtype=float), "t": ["a"] * 100})
    trained = stats(train)
    assert set(trained) == {"x"}  # text columns carry no mean

    same = s.drift_warnings(train, trained, 100, 100.0)
    assert same == []

    shifted = pd.DataFrame({"x": np.arange(100, dtype=float) + 1000})
    names = {w.name for w in s.drift_warnings(shifted, trained, 100, 100.0)}
    assert names == {"mean_shift"}

    holes = pd.DataFrame({"x": [np.nan] * 60 + list(np.arange(40, dtype=float))})
    assert "null_rate" in {w.name for w in s.drift_warnings(holes, trained, 100, 100.0)}

    counts = cast(Any, s.drift_warnings(train, trained, 400, 100.0))
    assert [w.name for w in counts] == ["entity_count"]


def test_the_summary_says_its_probabilities_are_not_calibrated() -> None:
    rows = pd.DataFrame({"score": [0.9, 0.5, 0.1]})
    out = s.summary(rows, 2)
    assert out["n_scored"] == 3 and out["top_k"] == 2 and out["score_at_k"] == 0.5
    assert out["expected_positives_in_top_k_uncalibrated"] == 1.4
    assert all("uncalibrated" in k for k in out if k.startswith("expected"))
