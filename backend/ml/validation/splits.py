"""One split contract for every experiment: which rows train, validate and test.

Tuning, early stopping, ensembles and keep/reject decisions only ever see the
train and validation rows. The test rows are scored once, at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

# Below this many rows a single validation split is too noisy; use inner CV instead.
SMALL_DATA_ROWS = 2000


class SplitPlan(BaseModel):
    strategy: Literal["holdout", "cv", "temporal"] = "holdout"
    seed: int = 42
    test_fraction: float = Field(0.2, gt=0, lt=1)
    # holdout: share of all rows used for validation
    val_fraction: float = Field(0.2, gt=0, lt=1)
    # cv: number of inner folds over the non-test rows
    inner_folds: Optional[int] = Field(None, ge=2)
    # temporal (filled in by the relational task work, #52)
    val_time: Optional[str] = None
    test_time: Optional[str] = None

    @classmethod
    def default_for(cls, n_rows: int, config: dict[str, Any] | None = None) -> "SplitPlan":
        """Holdout for normal-sized data, inner 5-fold CV under SMALL_DATA_ROWS rows."""
        config = dict(config or {})
        # Older experiments used random_state / test_size / val_size.
        if "random_state" in config:
            config.setdefault("seed", config.pop("random_state"))
        if "test_size" in config:
            config.setdefault("test_fraction", config.pop("test_size"))
        if "val_size" in config:
            config.setdefault("val_fraction", config.pop("val_size"))
        known = {k: v for k, v in config.items() if k in cls.model_fields}
        if "strategy" not in known and n_rows < SMALL_DATA_ROWS:
            known.update(strategy="cv", inner_folds=known.get("inner_folds") or 5)
        if known.get("strategy") == "cv" and not known.get("inner_folds"):
            known["inner_folds"] = 5
        return cls(**known)


@dataclass
class SplitIndices:
    """Positional row indices. For cv, `train` holds all non-test rows and `folds` the inner folds."""

    plan: SplitPlan
    stratified: bool
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.plan.strategy,
            "seed": self.plan.seed,
            "stratified": self.stratified,
            "n_train": int(len(self.train)),
            "n_val": int(len(self.val)),
            "n_test": int(len(self.test)),
            "inner_folds": len(self.folds) or None,
        }


def make_splits(y: pd.Series | np.ndarray, plan: SplitPlan) -> SplitIndices:
    """Turn a target column and a plan into disjoint train/val/test row indices."""
    if plan.strategy == "temporal":
        raise NotImplementedError("Temporal splits are built from the task spec (#52), not here.")
    y = pd.Series(np.asarray(y))
    rows = np.arange(len(y))
    stratified = bool(y.value_counts().min() >= max(5, plan.inner_folds or 0))
    strat = y if stratified else None
    rest, test = train_test_split(rows, test_size=plan.test_fraction, random_state=plan.seed, stratify=strat)
    rest, test = np.sort(rest), np.sort(test)

    if plan.strategy == "cv":
        splitter = (
            StratifiedKFold(n_splits=plan.inner_folds, shuffle=True, random_state=plan.seed)
            if stratified
            else KFold(n_splits=plan.inner_folds, shuffle=True, random_state=plan.seed)
        )
        folds = [(rest[a], rest[b]) for a, b in splitter.split(rest, y.iloc[rest])]
        return SplitIndices(plan, stratified, train=rest, val=np.array([], dtype=int), test=test, folds=folds)

    val_share = plan.val_fraction / (1 - plan.test_fraction)
    train, val = train_test_split(
        rest, test_size=val_share, random_state=plan.seed, stratify=y.iloc[rest] if stratified else None
    )
    return SplitIndices(plan, stratified, train=np.sort(train), val=np.sort(val), test=test)
