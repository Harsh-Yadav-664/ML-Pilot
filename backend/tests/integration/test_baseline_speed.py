"""The telecom baseline must be fast enough for a demo, and record what it changed."""

from __future__ import annotations

import time
from pathlib import Path

from ml.data.ingestion.csv_loader import CsvLoader
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec

SAMPLE = str(Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv")


async def test_telecom_baseline_under_60s_and_records_changes():
    loader = CsvLoader()
    executor = LocalExperimentExecutor(data_loader_func=lambda p: loader.load(p))
    # Same spec as POST /projects/{p}/experiments/baseline.
    spec = ExperimentSpec(
        id="speed",
        project_id="p",
        dataset_version=SAMPLE,
        hypothesis="Deterministic baseline without new features",
        change_description="Baseline run using RandomForestClassifier",
        model_name="RandomForestClassifier",
        parameters={
            "target_column": "Churn",
            "model_params": {"n_estimators": 50, "random_state": 42},
        },
        feature_set=[],
    )
    start = time.perf_counter()
    result = await executor.run(spec)
    elapsed = time.perf_counter() - start
    print(f"telecom baseline took {elapsed:.1f}s")

    assert result.status.value == "completed"
    assert elapsed < 60
    assert result.parameters["excluded_features"] == {
        "customerID": "id-like: one unique value per row"
    }
    assert result.parameters["numeric_coercion"] == {"TotalCharges": 11}
    assert result.parameters["tuning"] == "none: no search space for this model"
    assert 0 < result.metrics["f1"] < 1
