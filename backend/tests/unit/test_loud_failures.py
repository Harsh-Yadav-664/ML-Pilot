"""Failures are loud (AGENTS.md rule 8): every previously swallowed path now leaves a visible record."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import ml.experiments.executor as executor_module
import ml.metrics.classification as classification_module
from app.core.errors import describe, step_failed, unavailable
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from ml.metrics.classification import compute_classification_metrics


def _spec(model: str = "LGBMClassifier", **params) -> ExperimentSpec:
    return ExperimentSpec(
        id="loud",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name=model,
        parameters={"target_column": "target", **params},
        feature_set=[],
    )


def _frame(n: int = 400) -> pd.DataFrame:
    X, y = make_classification(n_samples=n, n_features=5, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)


async def _run(spec: ExperimentSpec):
    df = _frame()
    return await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)


async def test_optuna_failure_is_recorded_as_tuning_failed(monkeypatch):
    optuna = pytest.importorskip("optuna")

    def broken_study(*args, **kwargs):
        raise RuntimeError("study storage unavailable")

    monkeypatch.setattr(optuna, "create_study", broken_study)
    result = await _run(_spec(n_trials=2, ensemble=False))
    assert result.status.value == "completed"
    assert result.parameters["tuning"] == "failed (RuntimeError: study storage unavailable)"
    assert "tuning: used the given parameters" in result.parameters["skipped"]
    assert "best_params" not in result.parameters


async def test_ensemble_failure_is_recorded(monkeypatch):
    real_get_engine = executor_module.get_engine

    def get_engine(name):
        if name == "XGBClassifier":
            raise RuntimeError("xgboost crashed")
        return real_get_engine(name)

    monkeypatch.setattr(executor_module, "get_engine", get_engine)
    result = await _run(_spec("LogisticRegression"))
    assert result.status.value == "completed"
    assert result.parameters["ensemble_status"] == "failed (RuntimeError: xgboost crashed)"
    assert "ensemble_status: no ensemble metrics" in result.parameters["skipped"]
    assert not any(k.startswith("val_ensemble_") for k in result.metrics)


def test_roc_auc_failure_is_none_with_a_note(monkeypatch):
    def broken_auc(*args, **kwargs):
        raise ValueError("Input contains NaN")

    monkeypatch.setattr(classification_module, "roc_auc_score", broken_auc)
    y = np.array([0, 1, 0, 1])
    notes: dict[str, str] = {}
    metrics = compute_classification_metrics(y, y, np.full((4, 2), 0.5), notes=notes)
    assert metrics["roc_auc"] is None
    assert notes == {"roc_auc": "could not be computed: ValueError: Input contains NaN"}


def test_roc_auc_with_one_class_is_none_with_a_note():
    y = np.zeros(6, dtype=int)
    notes: dict[str, str] = {}
    metrics = compute_classification_metrics(y, y, np.tile([0.9, 0.1], (6, 1)), notes=notes)
    assert metrics["roc_auc"] is None and notes["roc_auc"] == "only one class present"
    assert metrics["accuracy"] == 1.0


async def test_metric_notes_reach_the_experiment_record(monkeypatch):
    def broken_auc(*args, **kwargs):
        raise ValueError("forced")

    monkeypatch.setattr(classification_module, "roc_auc_score", broken_auc)
    result = await _run(_spec("LogisticRegression", ensemble=False))
    assert result.metrics["roc_auc"] is None and result.metrics["test_roc_auc"] is None
    notes = result.parameters["metric_notes"]
    assert notes["roc_auc"] == notes["test_roc_auc"] == "could not be computed: ValueError: forced"


def test_helpers():
    record: dict = {}
    assert (
        step_failed(record, "step", KeyError("x"), skipped="nothing else")
        == "failed (KeyError: 'x')"
    )
    assert record == {"step": "failed (KeyError: 'x')", "skipped": ["step: nothing else"]}
    values, notes = {"a": 1.0}, {}
    unavailable(values, notes, "a", "why")
    assert values == {"a": None} and notes == {"a": "why"}
    assert describe(RuntimeError()) == "RuntimeError"
