"""Validation strategy implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ml.core.interfaces import ValidationSplit, ValidationStrategy
from ml.validation.splits import TemporalSplitPlan, make_temporal_splits


class StratifiedKFoldStrategy(ValidationStrategy):
    """Stratified k-fold cross-validation."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            splits.append(
                ValidationSplit(
                    train_indices=train_idx.tolist(), val_indices=val_idx.tolist(), fold=fold
                )
            )
        return splits

    def describe(self) -> str:
        return f"StratifiedKFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

    def get_config(self) -> dict[str, Any]:
        return {
            "strategy": "stratified_kfold",
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
        }


class KFoldStrategy(ValidationStrategy):
    """Standard k-fold cross-validation."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        X = df.drop(columns=[target_column])
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        splits = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            splits.append(
                ValidationSplit(
                    train_indices=train_idx.tolist(), val_indices=val_idx.tolist(), fold=fold
                )
            )
        return splits

    def describe(self) -> str:
        return f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

    def get_config(self) -> dict[str, Any]:
        return {
            "strategy": "kfold",
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
        }


class TimeSeriesSplitStrategy(ValidationStrategy):
    """Time-series cross-validation (no shuffling, respects temporal order)."""

    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        self.n_splits = n_splits
        self.gap = gap

    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        X = df.drop(columns=[target_column])
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        splits = []
        for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
            splits.append(
                ValidationSplit(
                    train_indices=train_idx.tolist(), val_indices=val_idx.tolist(), fold=fold
                )
            )
        return splits

    def describe(self) -> str:
        return f"TimeSeriesSplit(n_splits={self.n_splits}, gap={self.gap})"

    def get_config(self) -> dict[str, Any]:
        return {"strategy": "time_series_split", "n_splits": self.n_splits, "gap": self.gap}


class HoldoutStrategy(ValidationStrategy):
    """Simple holdout split (train/val)."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        import numpy as np
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )
        return [
            ValidationSplit(train_indices=train_idx.tolist(), val_indices=val_idx.tolist(), fold=0)
        ]

    def describe(self) -> str:
        return f"HoldoutStrategy(test_size={self.test_size})"

    def get_config(self) -> dict[str, Any]:
        return {
            "strategy": "holdout",
            "test_size": self.test_size,
            "random_state": self.random_state,
        }


class TemporalCutoffStrategy(ValidationStrategy):
    """Expanding-window folds over the training cutoffs of a relational task (#52).

    The frame holds one row per (entity, cutoff) with the columns of the label table. Folds
    train on earlier cutoffs whose label window ends before the fold's validation block. Rows of
    the validation and test periods are never in a fold.
    """

    def __init__(
        self,
        plan: TemporalSplitPlan,
        cutoff_column: str = "cutoff_time",
        window_end_column: str = "label_window_end",
    ) -> None:
        self.plan = plan
        self.cutoff_column = cutoff_column
        self.window_end_column = window_end_column

    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        result = make_temporal_splits(df[self.cutoff_column], df[self.window_end_column], self.plan)
        return [
            ValidationSplit(train_indices=f.train.tolist(), val_indices=f.val.tolist(), fold=i)
            for i, f in enumerate(result.folds)
        ]

    def describe(self) -> str:
        return (
            f"Temporal(val_from={self.plan.val_from:%Y-%m-%d}, "
            f"test_from={self.plan.test_from:%Y-%m-%d}, folds={self.plan.folds})"
        )

    def get_config(self) -> dict[str, Any]:
        return {"strategy": "temporal", **self.plan.model_dump(mode="json", exclude={"strategy"})}
