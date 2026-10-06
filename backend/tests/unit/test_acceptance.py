"""Keep/reject is decided by code: noise and constant features are rejected, real signal accepted."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.core.targets import TargetEncoder
from ml.data.preparation.feature_frame import prepare_feature_frame
from ml.experiments.acceptance import compare_feature_sets
from ml.experiments.executor import acceptance_pipeline
from ml.validation.splits import SplitPlan, make_splits

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture(scope="module")
def telecom_training_rows():
    df = pd.read_csv(SAMPLE)
    X, _ = prepare_feature_frame(df.drop(columns=["Churn"]))
    split = make_splits(df["Churn"], SplitPlan.default_for(len(df), {}))
    rows = np.sort(np.concatenate([split.train, split.val]))
    y_raw = df["Churn"].iloc[rows]
    y = TargetEncoder.fit(y_raw).transform(y_raw)
    return X.iloc[rows], y


@pytest.mark.parametrize("seed", range(10))
def test_constant_and_noise_features_are_rejected(telecom_training_rows, seed):
    X, y = telecom_training_rows
    rng = np.random.default_rng(seed)
    for column in (np.zeros(len(X)), rng.normal(size=len(X))):
        result = compare_feature_sets(X, X.assign(candidate=column), y, acceptance_pipeline, random_state=seed)
        assert not result.accepted, result.to_dict()


def test_planted_informative_feature_is_accepted(telecom_training_rows):
    X, y = telecom_training_rows
    noisy_target = y + np.random.default_rng(0).normal(scale=1.5, size=len(X))
    result = compare_feature_sets(X, X.assign(candidate=noisy_target), y, acceptance_pipeline)
    assert result.accepted
    assert result.mean_gain > result.margin
    assert result.ci95[0] > 0
    assert len(result.base_scores) == len(result.candidate_scores) == 15
