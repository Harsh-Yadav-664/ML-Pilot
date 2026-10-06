"""Model engines: one interface, LightGBM by default, no silent fallback for missing engines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.schema import ExperimentSpec
from ml.models.engines import (
    DEFAULT_ENGINE, ENGINES, EngineNotAvailable, ModelEngine, get_engine, installed_engines,
)

BUILT_IN = ["LGBMClassifier", "XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"]


def _frame(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "charges": rng.normal(70, 20, n),
        "contract": rng.choice(["monthly", "yearly", "two-year"], n),
    })
    X.loc[::17, "charges"] = np.nan
    y = ((X["contract"] == "monthly") & (X["tenure"] < 30)).astype(int).to_numpy()
    return X, y


def test_default_engine_is_lightgbm():
    assert DEFAULT_ENGINE == "LGBMClassifier"
    assert set(BUILT_IN) <= set(installed_engines())


@pytest.mark.parametrize("name", BUILT_IN)
def test_built_in_engines_follow_the_interface(name, tmp_path):
    X, y = _frame()
    engine = get_engine(name)
    assert isinstance(engine, ModelEngine)
    engine.fit(X, y, seed=0)
    prob = engine.predict_proba(X)
    assert prob.shape == (len(X), 2) and np.allclose(prob.sum(axis=1), 1)
    assert set(engine.predict(X)) <= {0, 1}
    assert set(engine.feature_importances()) <= set(X.columns)
    assert engine.describe()["name"] == name and engine.describe()["version"][0].isdigit()
    engine.save(str(tmp_path / "m.joblib"))
    assert np.allclose(type(engine).load(str(tmp_path / "m.joblib")).predict_proba(X), prob)


def test_lightgbm_gets_native_categories_not_one_hot():
    X, y = _frame()
    engine = get_engine("LGBMClassifier")
    engine.fit(X, y)
    encoded = engine.pipeline[:-1].transform(X)
    assert list(encoded.columns) == ["num__tenure", "num__charges", "cat__contract"]
    assert isinstance(encoded["cat__contract"].dtype, pd.CategoricalDtype)
    assert encoded["num__charges"].isna().any()  # LightGBM handles missing values itself
    # Categories unseen in training become missing, not an error.
    unseen = X.head(3).assign(contract="weekly")
    assert engine.predict_proba(unseen).shape == (3, 2)


@pytest.mark.parametrize("name", [n for n in ("AutoGluon", "TabICL") if n not in installed_engines()])
def test_missing_optional_engine_is_a_clear_error(name):
    with pytest.raises(EngineNotAvailable, match=f"'{name}' is not installed .*No other engine was used"):
        get_engine(name)


def test_unknown_engine_is_a_clear_error():
    with pytest.raises(EngineNotAvailable, match="Unknown engine 'CatBoost'"):
        get_engine("CatBoost")


async def test_executor_fails_loudly_for_missing_engine_and_records_engine_otherwise():
    X, y = _frame()
    df = X.assign(target=y)
    run = LocalExperimentExecutor(data_loader_func=lambda _: df.copy()).run
    missing = next((n for n in ENGINES if n not in installed_engines()), None)
    if missing:
        spec = ExperimentSpec(id="m", project_id="p", dataset_version="v", hypothesis="h", change_description="c",
                              model_name=missing, parameters={"target_column": "target"})
        with pytest.raises(EngineNotAvailable):
            await run(spec)
    spec = ExperimentSpec(id="ok", project_id="p", dataset_version="v", hypothesis="h", change_description="c",
                          model_name=DEFAULT_ENGINE, parameters={"target_column": "target", "n_trials": 2})
    result = await run(spec)
    assert result.parameters["engine"]["name"] == "LGBMClassifier"
    assert result.parameters["engine"]["version"] == get_engine("LGBMClassifier").version
