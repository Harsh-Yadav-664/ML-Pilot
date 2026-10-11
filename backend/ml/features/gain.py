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

from collections.abc import Sequence
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.experiments.acceptance import GainResult, decide
from ml.validation.splits import SplitError

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


class Fold(Protocol):
    """Rows to fit on and rows to score on: a temporal fold or a random one (single table)."""

    train: np.ndarray
    val: np.ndarray


class FoldScorer:
    """Scores feature sets on the folds of one split; the same folds every time.

    ``kind`` says what the folds are, and is recorded with every decision: ``temporal`` for the
    expanding-window folds of a relational task, ``random`` for the repeated stratified folds of
    a single-table task (which has no event time to order by).
    """

    def __init__(
        self,
        y: np.ndarray,
        folds: Sequence[Fold],
        *,
        train_rows: np.ndarray,
        seed: int = 42,
        params: dict[str, Any] | None = None,
        kind: str = "temporal",
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
        self.kind = kind
        self.fits = 0

    def score(self, frame: pd.DataFrame) -> list[float]:
        """PR-AUC on each fold's validation block, after fitting on its training rows."""
        scores = []
        for f in self.folds:
            y_train, y_val = self.y[f.train], self.y[f.val]
            if len(set(y_train)) < 2 or len(set(y_val)) < 2:
                start = getattr(f, "val_start", None)
                where = f"validation from {start:%Y-%m-%d}" if start is not None else "random fold"
                raise SplitError(
                    f"a fold has only one class ({where}): use fewer folds or more "
                    f"{'cutoffs' if start is not None else 'rows'}"
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
    kind: str = "temporal",
) -> GainResult:
    """Apply the paired rule (``ml.experiments.acceptance.decide``) to scores of two feature
    sets on the same folds."""
    rule = dict(rule or {})
    rule.pop("n_splits", None)  # the folds are the scorer's, not repeated K-fold
    rule.pop("n_repeats", None)
    result = decide(base_scores, candidate_scores, rule, metric=METRIC)
    for key in ("n_splits", "n_repeats"):
        result.rule.pop(key, None)
    result.rule.update(folds=len(base_scores), validation=kind)
    return result
