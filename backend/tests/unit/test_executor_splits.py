"""The split contract: disjoint, complete, reproducible, and recorded on the experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from ml.validation.splits import SMALL_DATA_ROWS, SplitPlan, make_splits


def _target(n: int, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).choice(["yes", "no"], size=n, p=[0.3, 0.7]))


@pytest.mark.parametrize("strategy", ["holdout", "cv"])
@pytest.mark.parametrize("n_rows", [300, 5000])
def test_indices_are_disjoint_and_cover_every_row(strategy, n_rows):
    split = make_splits(_target(n_rows), SplitPlan.default_for(n_rows, {"strategy": strategy}))
    parts = [set(split.train), set(split.val), set(split.test)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert parts[0] | parts[1] | parts[2] == set(range(n_rows))
    for fit_idx, val_idx in split.folds:
        assert not set(fit_idx) & set(val_idx)
        assert not (set(fit_idx) | set(val_idx)) & parts[2]
        assert set(fit_idx) | set(val_idx) == parts[0]


def test_same_seed_gives_identical_indices_and_other_seed_differs():
    y = _target(3000)
    a = make_splits(y, SplitPlan(seed=7))
    b = make_splits(y, SplitPlan(seed=7))
    c = make_splits(y, SplitPlan(seed=8))
    for name in ("train", "val", "test"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.test, c.test)


def test_small_data_defaults_to_inner_five_fold_cv():
    small = SplitPlan.default_for(SMALL_DATA_ROWS - 1)
    assert small.strategy == "cv" and small.inner_folds == 5
    assert SplitPlan.default_for(SMALL_DATA_ROWS).strategy == "holdout"
    split = make_splits(_target(500), small)
    assert len(split.folds) == 5 and len(split.val) == 0


def test_old_config_keys_are_still_understood():
    plan = SplitPlan.default_for(5000, {"random_state": 3, "test_size": 0.25, "val_size": 0.1})
    assert (plan.seed, plan.test_fraction, plan.val_fraction) == (3, 0.25, 0.1)


def test_temporal_split_is_not_faked():
    with pytest.raises(NotImplementedError):
        make_splits(_target(100), SplitPlan(strategy="temporal"))


async def test_split_plan_is_recorded_on_the_experiment():
    X, y = make_classification(n_samples=400, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)
    spec = ExperimentSpec(
        id="plan",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name="LogisticRegression",
        parameters={"target_column": "target"},
        feature_set=[],
    )
    result = await LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run(spec)
    assert result.status.value == "completed"
    assert result.parameters["split_plan"] == SplitPlan.default_for(400).model_dump()
    assert result.parameters["split"] == {
        "strategy": "cv",
        "seed": 42,
        "stratified": True,
        "n_train": 320,
        "n_val": 0,
        "n_test": 80,
        "inner_folds": 5,
    }
