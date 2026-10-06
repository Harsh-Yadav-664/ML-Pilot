"""#93: the decision threshold and the calibrator never see test labels or test rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

import ml.experiments.executor as executor_module
from ml.core.targets import TargetEncoder
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from ml.validation.splits import SplitPlan, make_splits


def _data() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=1200, n_features=6, n_informative=4, weights=[0.85], random_state=0
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = np.where(y == 1, "yes", "no")
    return df


@pytest.fixture
def spy(monkeypatch):
    """Record, in order, every fit of the calibrator/threshold and every model call."""
    events: list[tuple[str, object]] = []
    real_cal, real_thr = executor_module.fit_calibrator, executor_module.choose_threshold
    real_proba = Pipeline.predict_proba

    def fit_calibrator(y, prob):
        events.append(("fit_calibrator", np.asarray(y).copy()))
        return real_cal(y, prob)

    def choose_threshold(y, prob):
        events.append(("choose_threshold", np.asarray(y).copy()))
        return real_thr(y, prob)

    def predict_proba(self, X, *args, **kwargs):
        events.append(("predict_proba", frozenset(X.index)))
        return real_proba(self, X, *args, **kwargs)

    monkeypatch.setattr(executor_module, "fit_calibrator", fit_calibrator)
    monkeypatch.setattr(executor_module, "choose_threshold", choose_threshold)
    monkeypatch.setattr(Pipeline, "predict_proba", predict_proba)
    return events


@pytest.mark.parametrize("strategy", ["holdout", "cv"])
async def test_threshold_and_calibrator_are_fitted_without_test_labels(spy, strategy):
    df = _data()
    spec = ExperimentSpec(
        id="t",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name="LogisticRegression",
        parameters={"target_column": "target"},
        feature_set=[],
        validation_config={"strategy": strategy},
    )
    result = await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)
    assert result.status.value == "completed"

    plan = SplitPlan.default_for(len(df), {"strategy": strategy})
    split = make_splits(df["target"], plan)
    test_rows = frozenset(df.index[split.test])
    encoder = TargetEncoder.fit(df["target"].iloc[split.train])
    # The labels they were fitted on: validation labels (holdout) or out-of-fold training
    # labels (cv); in both cases the test labels are not among them.
    fit_rows = split.val if strategy == "holdout" else split.train
    expected = encoder.transform(df["target"].iloc[fit_rows])
    fits = [(name, y) for name, y in spy if name in ("fit_calibrator", "choose_threshold")]
    assert [name for name, _ in fits] == ["fit_calibrator", "choose_threshold"]
    for _, y in fits:
        assert np.array_equal(y, expected)

    # Both were fitted before the model ever predicted a test row.
    first_test = next(
        i for i, (n, rows) in enumerate(spy) if n == "predict_proba" and rows & test_rows
    )
    assert all(i < first_test for i, (n, _) in enumerate(spy) if n != "predict_proba")

    # ...and are recorded on the run, then applied unchanged to the test metrics.
    p = result.parameters
    assert p["calibration"]["fitted_on"] == "validation"
    assert p["threshold"]["chosen_on"] == "validation"
    for key in ("base_rate", "pr_auc", "lift_at_10pct", "brier", "ece", "f1_at_threshold"):
        assert result.metrics[f"val_{key}"] is not None
        assert result.metrics[f"test_{key}"] is not None
    assert result.metrics["test_base_rate"] == pytest.approx(
        (df["target"].iloc[split.test] == "yes").mean()
    )
