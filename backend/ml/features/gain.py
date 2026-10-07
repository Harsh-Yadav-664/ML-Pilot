"""Does a feature help? Decided in code, on time-ordered folds of the training rows (#58).

The rule is the paired one of ADR 0006: score the current feature set (the champion) and the
champion plus the candidate on the *same* folds, and accept only if the mean paired gain beats a
margin tied to the noise. For relational tasks the folds are the expanding-window folds of the
temporal split (``TemporalSplit.folds``): each fold trains on earlier cutoffs and validates on a
later block, with the gap rule that keeps label windows from overlapping. Only training rows are
ever used here. The validation rows were used for early stopping and the test rows are scored
once, at the end of a run, so neither takes part in a decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.experiments.acceptance import DEFAULT_RULE, GainResult
from ml.validation.splits import SplitError, TemporalFold

# Smaller and faster than the final model: it is fitted twice per fold for every candidate.
FOLD_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
METRIC = "pr_auc"
RULE_TEXT = "accept if mean paired gain > max(min_gain, std_multiplier * std of paired gains)"


class FoldScorer:
    """Scores feature sets on the folds of one split; the same folds every time."""

    def __init__(
        self,
        y: np.ndarray,
        folds: Sequence[TemporalFold],
        *,
        train_rows: np.ndarray,
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ) -> None:
        if not folds:
            raise SplitError("acceptance needs at least one temporal fold")
        allowed = set(train_rows.tolist())
        for f in folds:
            if not (set(f.train.tolist()) | set(f.val.tolist())) <= allowed:
                raise SplitError("a fold uses rows that are not training rows")
        self.y = np.asarray(y)
        self.folds = list(folds)
        self.seed = seed
        self.params = {**FOLD_PARAMS, **(params or {})}
        self.fits = 0

    def score(self, frame: pd.DataFrame) -> list[float]:
        """PR-AUC on each fold's validation block, after fitting on its training rows."""
        scores = []
        for f in self.folds:
            y_train, y_val = self.y[f.train], self.y[f.val]
            if len(set(y_train)) < 2 or len(set(y_val)) < 2:
                raise SplitError(
                    f"a fold has only one class (validation from {f.val_start:%Y-%m-%d}): "
                    "use fewer folds or more cutoffs"
                )
            model = lgb.LGBMClassifier(
                **self.params,
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
            )
            model.fit(frame.iloc[f.train], y_train)
            self.fits += 1
            prob = np.asarray(model.predict_proba(frame.iloc[f.val]))[:, 1]
            scores.append(float(average_precision_score(y_val, prob)))
        return scores


def compare(
    base_scores: Sequence[float],
    candidate_scores: Sequence[float],
    rule: dict[str, Any] | None = None,
) -> GainResult:
    """Apply the paired rule to scores of two feature sets on the same folds."""
    rule = {**DEFAULT_RULE, **(rule or {})}
    if len(base_scores) != len(candidate_scores):
        raise ValueError("both feature sets need a score on every fold")
    diffs = np.array(candidate_scores) - np.array(base_scores)
    mean_gain = float(diffs.mean())
    std_gain = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    half_width = 1.96 * std_gain / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    margin = max(rule["min_gain"], rule["std_multiplier"] * std_gain)
    return GainResult(
        metric=METRIC,
        base_scores=list(base_scores),
        candidate_scores=list(candidate_scores),
        mean_gain=mean_gain,
        std_gain=std_gain,
        ci95=(mean_gain - half_width, mean_gain + half_width),
        margin=margin,
        accepted=mean_gain > margin,
        rule={**rule, "description": RULE_TEXT, "folds": len(diffs), "validation": "temporal"},
    )
