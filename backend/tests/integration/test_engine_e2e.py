"""End to end on the telecom sample with one engine: baseline plus one candidate feature.

Runs with LightGBM (the default) in the normal suite. The optional CI job sets
MLPILOT_E2E_ENGINE=AutoGluon after installing autogluon.tabular.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"
ENGINE = os.environ.get("MLPILOT_E2E_ENGINE", "LGBMClassifier")
# AutoGluon: medium_quality preset with a time limit per fit.
MODEL_PARAMS = {"presets": "medium_quality", "time_limit": 60} if ENGINE == "AutoGluon" else {}


async def test_engine_runs_baseline_and_candidate_on_telecom(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)  # tuning is covered elsewhere
    df = pd.read_csv(SAMPLE)
    run = LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run
    common = dict(project_id="p", dataset_version=str(SAMPLE), hypothesis="h", change_description="c",
                  model_name=ENGINE)
    base = await run(ExperimentSpec(id="base", **common, parameters={
        "target_column": "Churn", "model_params": dict(MODEL_PARAMS), "ensemble": False}))
    cand = await run(ExperimentSpec(id="cand", **common, parameters={
        "target_column": "Churn", "model_params": dict(MODEL_PARAMS), "ensemble": False,
        "feature_name": "charge_x_tenure", "formula": "MonthlyCharges * tenure"}))
    for result in (base, cand):
        assert result.status.value == "completed"
        assert result.parameters["engine"]["name"] == ENGINE
        assert 0.5 < result.metrics["val_roc_auc"] <= 1.0 and 0.5 < result.metrics["test_roc_auc"] <= 1.0
    assert "accepted" in cand.parameters["acceptance"]
    print(f"\n{ENGINE}: baseline test AUC {base.metrics['test_roc_auc']:.3f}, "
          f"candidate test AUC {cand.metrics['test_roc_auc']:.3f}")
