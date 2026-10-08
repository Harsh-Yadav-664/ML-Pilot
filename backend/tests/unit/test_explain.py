"""TreeSHAP values and the retrieval-first answers (#63), without a database."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.reports import explain


def fitted() -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "orders__count_30d": rng.poisson(3, n).astype(float),
            "refunds__count_90d": rng.poisson(1, n).astype(float),
            "noise": rng.normal(size=n),
            "segment": pd.Series(rng.choice(["a", "b", "c"], n)).astype("category"),
        }
    )
    logit = -0.6 * frame["orders__count_30d"] + 1.2 * frame["refunds__count_90d"] + 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = lgb.LGBMClassifier(n_estimators=40, random_state=0, verbose=-1, deterministic=True)
    model.fit(frame, y)
    return model, frame


def test_contributions_add_up_to_the_raw_score() -> None:
    model, frame = fitted()
    values, base = explain.contributions(model, frame)
    raw = model.predict(frame, raw_score=True)
    assert values.shape == frame.shape
    np.testing.assert_allclose(values.sum(axis=1) + base, raw, atol=1e-9)


def test_the_summary_ranks_the_features_that_drive_the_model() -> None:
    model, frame = fitted()
    summary = explain.shap_summary(model, frame, np.arange(len(frame)))
    ranked = sorted(summary["mean_abs"], key=lambda k: -summary["mean_abs"][k])
    assert set(ranked[:2]) == {"orders__count_30d", "refunds__count_90d"}
    assert abs(sum(summary["share"].values()) - 1.0) < 1e-9 and summary["n_rows"] == len(frame)


def test_top_reasons_are_the_largest_pushes_of_the_row() -> None:
    model, frame = fitted()
    values, _ = explain.contributions(model, frame.iloc[:5])
    for row, reasons in zip(values, explain.top_reasons(model, frame.iloc[:5], 3), strict=True):
        assert len(reasons) == 3
        assert [abs(v) for _, v in reasons] == sorted((abs(v) for _, v in reasons), reverse=True)
        assert abs(reasons[0][1]) == np.abs(row).max()


CORPUS = explain.build_corpus(
    {"status": "completed", "validation": {"pr_auc": 0.6329}, "test": {"pr_auc": 0.4693}},
    [
        {"name": "orders__count_30d", "status": "accepted", "gain": {"mean_gain": 0.0123}},
        {"name": "refund_share", "status": "rejected_gain", "gain": {"mean_gain": -0.004}},
        {"name": "leaky_sql", "status": "rejected_guard", "reasons": ["reads the future"]},
    ],
    {"method": "TreeSHAP", "n_rows": 500, "mean_abs": {"orders__count_30d": 0.8}, "share": {}},
)


def test_a_named_feature_is_answered_from_its_record() -> None:
    found = explain.retrieve("Why was refund_share rejected?", CORPUS)
    assert [r.id for r in found] == ["feature:refund_share"]
    assert "-0.0040" in explain.records_answer(found)


def test_a_feature_that_was_never_proposed_matches_nothing() -> None:
    assert explain.retrieve("what about weekend_magic_feature?", CORPUS) == []


def test_intents_find_the_run_and_the_rejected_features() -> None:
    assert [r.id for r in explain.retrieve("what is the test score?", CORPUS)] == ["run"]
    ids = [r.id for r in explain.retrieve("which features were rejected?", CORPUS)]
    assert ids == ["feature:refund_share", "feature:leaky_sql"]


def test_wording_is_checked_for_numbers_and_names_outside_the_records() -> None:
    records = explain.retrieve("test score and importance", CORPUS)
    assert explain.violations("The test PR-AUC is 0.4693, see [run].", records) == []
    assert explain.violations("The test PR-AUC is 0.9100.", records) == ["0.9100"]
    assert explain.violations("orders__count_7d matters most.", records) == ["orders__count_7d"]
