"""The test split is scored once, after tuning, and never used for fitting or tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from ml.experiments.executor import LocalExperimentExecutor, split_train_val_test
from ml.experiments.schema import ExperimentSpec


def _data() -> pd.DataFrame:
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["plan"] = np.random.default_rng(0).choice(["a", "b", "c"], size=400)
    df["target"] = np.where(y == 1, "yes", "no")
    return df


@pytest.fixture
def spy(monkeypatch):
    calls: list[tuple[str, frozenset]] = []
    real = {name: getattr(Pipeline, name) for name in ("fit", "predict", "predict_proba")}

    def wrap(name):
        def method(self, X, *args, **kwargs):
            calls.append((name, frozenset(X.index)))
            return real[name](self, X, *args, **kwargs)
        return method

    for name in real:
        monkeypatch.setattr(Pipeline, name, wrap(name))
    return calls


@pytest.mark.parametrize("model_name", ["XGBClassifier", "RandomForestClassifier"])
async def test_test_split_is_predicted_exactly_once_after_tuning(spy, model_name):
    df = _data()
    spec = ExperimentSpec(
        id="split", project_id="p", dataset_version="v", hypothesis="h", change_description="c",
        model_name=model_name,
        parameters={"target_column": "target", "n_trials": 3, "model_params": {"n_estimators": 20}},
        feature_set=[],
    )
    result = await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)
    assert result.status.value == "completed"

    _, _, X_test, *_ = split_train_val_test(df.drop(columns=["target"]), df["target"], {})
    test_rows = frozenset(X_test.index)

    on_test = [i for i, (name, rows) in enumerate(spy) if rows & test_rows]
    # Exactly one model call touches test rows: the final predict_proba...
    assert len(on_test) == 1
    name, rows = spy[on_test[0]]
    assert name == "predict_proba" and rows == test_rows
    # ...and it is the last fit/predict call involving the experiment's model, after all tuning.
    assert on_test[0] > max(i for i, (n, _) in enumerate(spy) if n == "fit" and not (spy[i][1] & test_rows))
    # No fit ever includes a test row.
    assert all(not (rows & test_rows) for n, rows in spy if n == "fit")


async def test_record_has_separate_val_and_test_metrics_and_split():
    df = _data()
    spec = ExperimentSpec(
        id="metrics", project_id="p", dataset_version="v", hypothesis="h", change_description="c",
        model_name="LGBMClassifier", parameters={"target_column": "target", "n_trials": 2}, feature_set=[],
    )
    result = await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)
    m = result.metrics
    for k in ("f1", "accuracy", "precision", "recall", "roc_auc"):
        assert f"val_{k}" in m and f"test_{k}" in m
        assert m[k] == m[f"test_{k}"]
    assert not any(k.startswith("ensemble_") for k in m)  # ensemble is validation-only now
    split = result.parameters["split"]
    assert (split["n_train"], split["n_val"], split["n_test"]) == (240, 80, 80)
    assert split["random_state"] == 42 and split["stratified"] is True
    assert "validation" in result.parameters["tuning"]
