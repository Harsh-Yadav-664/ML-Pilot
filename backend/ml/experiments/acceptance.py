"""Decide in code whether a candidate feature really helps.

A feature is accepted only when its gain over the current feature set beats
the noise of the comparison: both feature sets are scored on the same
repeated K-fold splits (training rows only, never the test split), and the
mean paired gain must exceed a margin.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

DEFAULT_RULE: dict[str, Any] = {
    "n_splits": 5,
    "n_repeats": 3,
    "min_gain": 0.002,
    "std_multiplier": 1.0,
}


@dataclass
class GainResult:
    metric: str
    base_scores: list[float]
    candidate_scores: list[float]
    mean_gain: float
    std_gain: float
    ci95: tuple[float, float]
    margin: float
    accepted: bool
    rule: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ci95"] = list(self.ci95)
        return out


def _score(y_true: np.ndarray, prob: np.ndarray) -> float:
    if prob.shape[1] == 2:
        return float(roc_auc_score(y_true, prob[:, 1]))
    return float(f1_score(y_true, prob.argmax(axis=1), average="weighted"))


def compare_feature_sets(
    X_base: pd.DataFrame,
    X_candidate: pd.DataFrame,
    y: np.ndarray,
    make_pipeline: Callable[[pd.DataFrame], Pipeline],
    rule: dict[str, Any] | None = None,
    random_state: int = 42,
) -> GainResult:
    """Score both feature sets on identical repeated stratified folds and apply the rule.

    `make_pipeline(X)` must return an unfitted pipeline suited to X's columns.
    `y` must be the encoded labels for the rows of X (training rows only).
    """
    rule = {**DEFAULT_RULE, **(rule or {})}
    y = np.asarray(y)
    binary = len(np.unique(y)) == 2
    folds = RepeatedStratifiedKFold(
        n_splits=rule["n_splits"], n_repeats=rule["n_repeats"], random_state=random_state
    )
    base_template = make_pipeline(X_base)
    cand_template = make_pipeline(X_candidate)
    base_scores: list[float] = []
    cand_scores: list[float] = []
    for train_idx, val_idx in folds.split(X_base, y):
        for template, X, scores in (
            (base_template, X_base, base_scores),
            (cand_template, X_candidate, cand_scores),
        ):
            pipe = clone(template)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            scores.append(_score(y[val_idx], pipe.predict_proba(X.iloc[val_idx])))

    diffs = np.array(cand_scores) - np.array(base_scores)
    mean_gain = float(diffs.mean())
    std_gain = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    half_width = 1.96 * std_gain / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    margin = max(rule["min_gain"], rule["std_multiplier"] * std_gain)
    return GainResult(
        metric="roc_auc" if binary else "f1_weighted",
        base_scores=base_scores,
        candidate_scores=cand_scores,
        mean_gain=mean_gain,
        std_gain=std_gain,
        ci95=(mean_gain - half_width, mean_gain + half_width),
        margin=margin,
        accepted=mean_gain > margin,
        rule={
            **rule,
            "description": "accept if mean paired gain > max(min_gain, std_multiplier * std of paired gains)",
        },
    )
