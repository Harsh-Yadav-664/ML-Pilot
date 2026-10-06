import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.baselines.classification import ClassificationBaseline
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec


def generate_mock_data():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y

    # Introduce some NaNs to test robust preprocessing
    df.loc[0:10, "feature_0"] = np.nan

    # Introduce a categorical column
    df["cat_feature"] = np.random.choice(["A", "B", "C"], size=500)

    return df


def test_classification_baseline():
    df = generate_mock_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    baseline = ClassificationBaseline(cv_folds=3)
    results = baseline.run(X_train=X, y_train=y)

    assert len(results) == len(baseline.models)

    # Check that at least one model trained successfully and returned metrics
    successes = [r for r in results if not r.error]
    assert len(successes) > 0

    rf_result = next((r for r in successes if r.model_name == "RandomForestClassifier"), None)
    assert rf_result is not None
    assert "f1" in rf_result.metrics
    assert "accuracy" in rf_result.metrics
    assert rf_result.metrics["f1"] > 0.5  # Basic sanity check


@pytest.mark.asyncio
async def test_local_experiment_executor():
    df = generate_mock_data()

    def mock_loader(version: str) -> pd.DataFrame:
        if version == "v1":
            return df
        raise ValueError("Unknown version")

    executor = LocalExperimentExecutor(data_loader_func=mock_loader)

    spec = ExperimentSpec(
        id="exp_001",
        project_id="proj_001",
        dataset_version="v1",
        hypothesis="Test pipeline execution",
        change_description="Baseline run",
        model_name="RandomForestClassifier",
        parameters={"target_column": "target", "model_params": {"n_estimators": 10}},
        validation_config={"test_size": 0.2, "random_state": 42},
        feature_set=[],  # use all
    )

    result = await executor.run(spec)

    assert result.status.value == "completed"
    assert result.metrics is not None
    assert "f1" in result.metrics
    assert "accuracy" in result.metrics
    assert "precision" in result.metrics
    assert "recall" in result.metrics
    assert "roc_auc" in result.metrics

    assert result.runtime_seconds > 0
