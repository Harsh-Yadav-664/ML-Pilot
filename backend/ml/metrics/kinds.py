"""What differs between the three task types of a relational run: binary, regression, multiclass.

One ``TaskKind`` says how the label is encoded, which LightGBM model is fitted, what it predicts,
which metric decides whether a feature helps (``metric``, scored per time-ordered fold) and which
metrics are reported for the validation and test rows. Everything else in a run (the split, the
gain rule, the loop, the test rows scored once) is shared.

* binary: PR-AUC, higher is better. The gain of a feature is the plain difference.
* regression: MAE, lower is better. The gain is the *relative* reduction of the fold's MAE
  ``(base - candidate) / base``, so the margin of the rule means the same for dollars and for
  counts. A target that is never negative and is zero in more than ``TWEEDIE_ZERO_SHARE`` of the
  training rows (spend: most customers buy nothing) is fitted with the Tweedie objective; the
  choice reads training rows only and is recorded with the model parameters.
* multiclass: log loss, lower is better, relative reduction. The class mapping is built from the
  training rows only (sorted class names, index = position) and stored with the run. A validation
  or test row whose class the training rows never had cannot be scored; it is left out and counted
  (``n_unseen_class``), never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.metrics.ranking import DEFAULT_FRACTIONS, captured_value_metrics, ranking_metrics

Task = Literal["binary", "regression", "multiclass"]
TWEEDIE_ZERO_SHARE = 0.3
MAX_CLASSES = 30
UNSEEN = -1  # the code of a class the training rows do not have
_EPS = 1e-15
METRIC_NAMES = {"pr_auc": "PR-AUC", "mae": "MAE", "log_loss": "log loss"}


@dataclass(frozen=True)
class TaskKind:
    task: Task = "binary"
    objective: str = "binary"
    classes: tuple[str, ...] = ()  # multiclass: the class mapping, index = position

    # -- building one from the training labels ------------------------------------------------
    @classmethod
    def fit(cls, task: Task, train_labels: pd.Series) -> TaskKind:
        """The kind for a task, decided on the training rows' labels only."""
        if task == "binary":
            return cls()
        if task == "regression":
            y = train_labels.to_numpy(dtype="float64")
            zero_inflated = bool(len(y) and y.min() >= 0 and (y == 0).mean() > TWEEDIE_ZERO_SHARE)
            return cls("regression", "tweedie" if zero_inflated else "regression")
        classes = tuple(sorted(train_labels.astype(str).unique()))
        return cls("multiclass", "multiclass", classes)

    def encode(self, labels: pd.Series) -> np.ndarray:
        if self.task == "binary":
            return labels.astype(int).to_numpy()
        if self.task == "regression":
            return labels.to_numpy(dtype="float64")
        index = {c: i for i, c in enumerate(self.classes)}
        return labels.astype(str).map(index).fillna(UNSEEN).astype(int).to_numpy()

    # -- the model ----------------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return "LGBMRegressor" if self.task == "regression" else "LGBMClassifier"

    @property
    def eval_metric(self) -> str:
        return {"binary": "average_precision", "regression": "l1", "multiclass": "multi_logloss"}[
            self.task
        ]

    def estimator(self, params: dict[str, Any], seed: int) -> Any:
        common = {**params, "random_state": seed, "deterministic": True, "force_row_wise": True}
        common["verbose"] = -1
        if self.task == "regression":
            return lgb.LGBMRegressor(objective=self.objective, **common)
        if self.task == "multiclass":
            return lgb.LGBMClassifier(objective="multiclass", **common)
        return lgb.LGBMClassifier(**common)

    def predict(self, model: Any, x: pd.DataFrame) -> np.ndarray:
        """Binary: P(label 1), shape (n,). Regression: the value, (n,). Multiclass: (n, classes)."""
        if self.task == "regression":
            return np.asarray(model.predict(x), dtype="float64")
        proba = np.asarray(model.predict_proba(x), dtype="float64")
        if self.task == "binary":
            return proba[:, 1]
        if proba.shape[1] == len(self.classes):
            return proba
        # a fold's training rows may lack a class: it gets (almost) no probability
        full = np.full((len(proba), len(self.classes)), _EPS)
        full[:, np.asarray(model.classes_, dtype=int)] = proba
        return full / full.sum(axis=1, keepdims=True)

    # -- the metric that decides, per fold ------------------------------------------------------
    @property
    def metric(self) -> str:
        return {"binary": "pr_auc", "regression": "mae", "multiclass": "log_loss"}[self.task]

    @property
    def metric_label(self) -> str:
        return METRIC_NAMES[self.metric]

    @property
    def higher_is_better(self) -> bool:
        return self.task == "binary"

    @property
    def gain_text(self) -> str:
        """What a gain is a gain in: ``PR-AUC``, or ``relative reduction of MAE`` / ``of log loss``."""
        return (
            self.metric_label
            if self.higher_is_better
            else f"relative reduction of {self.metric_label}"
        )

    def fold_score(self, y: np.ndarray, pred: np.ndarray) -> float:
        if self.task == "binary":
            return float(average_precision_score(y, pred))
        if self.task == "regression":
            return float(mean_absolute_error(y, pred))
        return float(log_loss(y, pred, labels=list(range(len(self.classes)))))

    # -- the metrics that are reported ----------------------------------------------------------
    def evaluate(self, y: np.ndarray, pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Validation or test metrics. ``reference`` is the training rows' labels, for the trivial
        baseline (the training median, or the training class shares)."""
        if self.task == "binary":
            return ranking_metrics(y, pred, absolute_k=(100,))
        if self.task == "regression":
            out = {
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "r2": float(r2_score(y, pred)),
                "trivial_mae": float(np.abs(y - np.median(reference)).mean()),
            }
            return {**out, **captured_value_metrics(y, pred, DEFAULT_FRACTIONS, (100,))}
        known = y != UNSEEN
        y, pred = y[known], pred[known]
        if not len(y):
            raise ValueError("no row has a class the training rows have")
        k = len(self.classes)
        shares = np.bincount(reference[reference != UNSEEN], minlength=k).astype("float64")
        prior = np.tile(shares / shares.sum(), (len(y), 1))
        return {
            "macro_f1": float(f1_score(y, pred.argmax(axis=1), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred.argmax(axis=1))),
            "log_loss": self.fold_score(y, pred),
            "trivial_log_loss": float(log_loss(y, np.clip(prior, _EPS, 1), labels=list(range(k)))),
            "n_unseen_class": float(known.size - known.sum()),
        }

    def headline(self, metrics: dict[str, float]) -> dict[str, float | str]:
        """The one number a run is judged by and the trivial baseline it must beat."""
        trivial = {
            "binary": "base_rate",
            "regression": "trivial_mae",
            "multiclass": "trivial_log_loss",
        }
        return {
            "metric": self.metric,
            "value": metrics[self.metric],
            "trivial": metrics[trivial[self.task]],
        }
