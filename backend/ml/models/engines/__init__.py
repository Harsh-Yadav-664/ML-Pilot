"""Model engines: one interface for every model family MLPilot can train.

MLPilot does not try to out-fit AutoML libraries; it delegates fitting to an
engine and owns the data, features and validation around it. An engine says how
to build its estimator, which preprocessing it needs (LightGBM takes categories
natively; linear models need one-hot and scaling), what Optuna may tune, and
which package version produced the model.

Optional engines (AutoGluon, TabICL) import their package lazily. Asking for one
that isn't installed raises EngineNotAvailable; there is no silent fallback.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import tempfile
from typing import Any, ClassVar, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.core.registry import ProviderRegistry
from ml.data.preparation.feature_frame import ONEHOT_MAX_CATEGORIES

DEFAULT_ENGINE = "LGBMClassifier"


class EngineNotAvailable(ValueError):
    """The requested engine is unknown, or its package is not installed."""


@runtime_checkable
class ModelEngine(Protocol):
    name: str
    version: str

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, params=None, seed: int = 0
    ) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importances(self) -> dict[str, float]: ...
    def search_space(self) -> dict | None: ...
    def save(self, path: str) -> None: ...


# ── preprocessing ────────────────────────────────────────────────────────────


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = list(X.select_dtypes(include=["number", "bool"]).columns)
    return numeric, [c for c in X.columns if c not in numeric]


def one_hot_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute and scale numbers; impute and one-hot encode the rest (capped)."""
    numeric, categorical = _split_columns(X)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=ONEHOT_MAX_CATEGORIES,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


class ToCategory(BaseEstimator, TransformerMixin):
    """Text columns -> pandas categories with the categories seen in training (unseen -> NaN)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {
            c: pd.Index(X[c].dropna().astype("string").unique()).sort_values() for c in X.columns
        }
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, cats in self.categories_.items():
            values = X[c].astype("string")
            out[c] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def native_category_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numbers as they are (NaN kept); text as pandas categories for engines that split on them."""
    numeric, categorical = _split_columns(X)
    return ColumnTransformer(
        [("num", "passthrough", numeric), ("cat", ToCategory(), categorical)],
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")


def identity_preprocessor(X: pd.DataFrame) -> FunctionTransformer:
    """Raw columns for engines that do their own preprocessing."""
    return FunctionTransformer(feature_names_out="one-to-one")


# ── engines ──────────────────────────────────────────────────────────────────

_TREE_SPACE = {
    "n_estimators": ("int", 50, 300),
    "max_depth": ("int", 3, 8),
    "learning_rate": ("float", 0.01, 0.3),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.6, 1.0),
}


class SklearnEngine:
    """An engine whose model is an sklearn-compatible estimator inside MLPilot's pipeline."""

    name = "base"
    package = "scikit-learn"
    module = "sklearn"
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.columns: list[str] = []

    # what an engine defines
    def estimator(self, params: dict[str, Any]) -> BaseEstimator:
        raise NotImplementedError

    def preprocessor(self, X: pd.DataFrame):
        return one_hot_preprocessor(X)

    def search_space(self) -> dict | None:
        return None

    # shared behaviour
    @classmethod
    def available(cls) -> bool:
        try:
            return importlib.util.find_spec(cls.module) is not None
        except ModuleNotFoundError:  # parent package of a dotted module is missing
            return False

    @property
    def version(self) -> str:
        return importlib.metadata.version(self.package)

    def make_estimator(self, params: dict[str, Any] | None = None, seed: int = 0) -> BaseEstimator:
        merged = {**self.defaults, **(params or {})}
        est = self.estimator(merged)
        if "random_state" in est.get_params():
            est.set_params(random_state=seed)
        return est

    def make_pipeline(
        self,
        X: pd.DataFrame,
        params: dict[str, Any] | None = None,
        seed: int = 0,
        preprocessor=None,
    ) -> Pipeline:
        prep = clone(preprocessor) if preprocessor is not None else self.preprocessor(X)
        return Pipeline([("preprocessor", prep), ("classifier", self.make_estimator(params, seed))])

    def fit(self, X, y, X_val=None, y_val=None, params=None, seed=0) -> None:
        self.columns = list(X.columns)
        self.pipeline = self.make_pipeline(X, params, seed)
        self.pipeline.fit(X, y)

    def _fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} engine is not fitted; call fit() first")
        return self.pipeline

    def predict_proba(self, X) -> np.ndarray:
        return self._fitted().predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self._fitted().predict(X)

    def feature_importances(self) -> dict[str, float]:
        from ml.metrics.importance import builtin_importances

        return builtin_importances(self.pipeline, self.columns, top_n=len(self.columns))

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> SklearnEngine:
        engine = cls()
        engine.pipeline = joblib.load(path)
        return engine

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


class LightGBMEngine(SklearnEngine):
    name, package, module = "LGBMClassifier", "lightgbm", "lightgbm"
    defaults: ClassVar[dict[str, Any]] = {"verbose": -1}

    def estimator(self, params):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)

    def preprocessor(self, X):
        return native_category_preprocessor(X)

    def search_space(self):
        return dict(_TREE_SPACE)


class XGBoostEngine(SklearnEngine):
    name, package, module = "XGBClassifier", "xgboost", "xgboost"

    def estimator(self, params):
        from xgboost import XGBClassifier

        return XGBClassifier(**params)

    def search_space(self):
        return dict(_TREE_SPACE)


class RandomForestEngine(SklearnEngine):
    name, module = "RandomForestClassifier", "sklearn"

    def estimator(self, params):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)


class GradientBoostingEngine(SklearnEngine):
    name, module = "GradientBoostingClassifier", "sklearn"

    def estimator(self, params):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(**params)


class LogisticRegressionEngine(SklearnEngine):
    name, module = "LogisticRegression", "sklearn"
    defaults: ClassVar[dict[str, Any]] = {"max_iter": 1000}

    def estimator(self, params):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**params)


class AutoGluonClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style wrapper around AutoGluon's TabularPredictor (imported on fit)."""

    def __init__(
        self, presets: str = "medium_quality", time_limit: int = 60, random_state: int = 0
    ):
        self.presets = presets
        self.time_limit = time_limit
        self.random_state = random_state

    def fit(self, X, y):
        from autogluon.tabular import TabularPredictor

        data = pd.DataFrame(X).copy()
        data["__label__"] = np.asarray(y)
        self.classes_ = np.unique(np.asarray(y))
        self.predictor_ = TabularPredictor(
            label="__label__", path=tempfile.mkdtemp(prefix="mlpilot-ag-"), verbosity=0
        ).fit(
            data,
            presets=self.presets,
            time_limit=self.time_limit,
            ag_args_fit={"random_seed": self.random_state},
        )
        return self

    def predict_proba(self, X):
        prob = self.predictor_.predict_proba(pd.DataFrame(X))
        return prob[list(self.classes_)].to_numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class AutoGluonEngine(SklearnEngine):
    name, package, module = "AutoGluon", "autogluon.tabular", "autogluon.tabular"
    defaults: ClassVar[dict[str, Any]] = {"presets": "medium_quality", "time_limit": 60}

    def estimator(self, params):
        return AutoGluonClassifier(**params)

    def preprocessor(self, X):
        return identity_preprocessor(X)


class TabICLEngine(SklearnEngine):
    name, package, module = "TabICL", "tabicl", "tabicl"

    def estimator(self, params):
        from tabicl import TabICLClassifier

        return TabICLClassifier(**params)


ENGINES: dict[str, type[SklearnEngine]] = {
    cls.name: cls
    for cls in (
        LightGBMEngine,
        XGBoostEngine,
        RandomForestEngine,
        GradientBoostingEngine,
        LogisticRegressionEngine,
        AutoGluonEngine,
        TabICLEngine,
    )
}
OPTIONAL_ENGINES = {"AutoGluon": "pip install autogluon.tabular", "TabICL": "pip install tabicl"}

engine_registry = ProviderRegistry()
for _name, _cls in ENGINES.items():
    engine_registry.register_class(_name, _cls)


def get_engine(name: str) -> SklearnEngine:
    """A fresh engine by name. Raises EngineNotAvailable for unknown or uninstalled engines."""
    cls = ENGINES.get(name)
    if cls is None:
        raise EngineNotAvailable(f"Unknown engine {name!r}. Available: {sorted(ENGINES)}")
    if not cls.available():
        hint = OPTIONAL_ENGINES.get(name, f"install the '{cls.package}' package")
        raise EngineNotAvailable(
            f"Engine {name!r} is not installed ({hint}). No other engine was used instead."
        )
    return cls()


def installed_engines() -> list[str]:
    return [name for name, cls in ENGINES.items() if cls.available()]
