"""The baseline of a single-table task (CSV, Parquet): the start of the one run loop (#151).

A table with a target column is a one-table database, and its task runs through the same loop
as a relational one (``ml.agents.run_loop``). Only the starting point differs, and this module
builds it:

* **Preprocessing** is the existing one (``prepare_feature_frame``: id-like columns dropped,
  numbers stored as text converted). Text columns become LightGBM categories.
* **Split** is random, stratified and fixed by the seed (the table has no event time; AGENTS.md
  allows random splits for single-table tasks): train, validation and test rows. The target is
  encoded from the training rows only.
* **Acceptance folds** are repeated stratified K-fold over the train and validation rows (never
  the test rows), 5 x 3 by default as before: the same folds score the champion and every
  candidate, and the paired rule of ``ml.experiments.acceptance.decide`` accepts or rejects.
* **Model** is LightGBM, stopped early on the validation rows, as in the relational baseline.

The result is the same ``BaselineResult`` a relational baseline gives, so the loop, the
acceptance rule and the one scoring of the test rows are shared. Binary targets only: a target
with more classes raises ``BaselineError`` (the older formula loop handled those; see
``docs/features.md``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from ml.core.targets import TargetEncoder
from ml.data.preparation.feature_frame import prepare_feature_frame
from ml.experiments.acceptance import DEFAULT_RULE
from ml.features.baseline import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    BaselineError,
    BaselineResult,
    KeptFeature,
    fit_validated,
)
from ml.features.dfs import Candidate
from ml.validation.splits import SplitError, SplitPlan, make_splits

SEED = 42


@dataclass
class TableFold:
    train: np.ndarray
    val: np.ndarray


@dataclass
class TableSplit:
    """Positional row indices of a random split of one table."""

    plan: SplitPlan
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    folds: list[TableFold]
    pool: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=int))

    @property
    def fold_rows(self) -> np.ndarray:
        """The rows the acceptance folds are drawn from: train and validation, never test."""
        return self.pool

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": "random_holdout",
            "seed": self.plan.seed,
            "stratified": True,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "acceptance_folds": len(self.folds),
        }


@dataclass
class TableTask:
    """What a formula proposer and the recorder need besides the baseline result."""

    target_column: str
    features: pd.DataFrame = field(repr=False)  # prepared columns, original dtypes, row order
    y: np.ndarray = field(repr=False)  # encoded labels, 1 is the positive class
    encoder: TargetEncoder
    report: dict[str, Any]  # excluded_features, numeric_coercion
    split: TableSplit


def model_frame(features: pd.DataFrame) -> pd.DataFrame:
    """The columns as a model reads them: numbers as floats, text as categories."""
    out: dict[str, pd.Series] = {}
    for name in features.columns:
        col = features[name]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            out[str(name)] = col.astype("float64")
        elif pd.api.types.is_datetime64_any_dtype(col):
            days = (col - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
            out[str(name)] = days.astype("float64")
        else:
            out[str(name)] = col.astype("string").astype("category")
    return pd.DataFrame(out, index=features.index)


def build_table_baseline(
    df: pd.DataFrame,
    target_column: str,
    *,
    seed: int = SEED,
    positive_class: Any | None = None,
    rule: dict[str, Any] | None = None,
) -> tuple[BaselineResult, TableTask]:
    """The prepared table, its random split and folds, and the LightGBM baseline on them."""
    started = time.monotonic()
    if target_column not in df.columns:
        raise BaselineError(f"Target column {target_column!r} not found in the dataset")
    df = df.reset_index(drop=True)
    target = df[target_column]
    if target.isna().any():
        raise BaselineError(
            f"The target column {target_column!r} has {int(target.isna().sum())} missing values; "
            "rows without a label cannot train or test a model"
        )
    features, report = prepare_feature_frame(df.drop(columns=[target_column]))
    if features.shape[1] == 0:
        raise BaselineError("no feature column is left after the id-like columns are dropped")
    features = features.reset_index(drop=True)

    # the older single-table path also split at random by default; the loop needs validation
    # rows for early stopping, so a small table is split the same way (holdout) rather than by CV
    plan = SplitPlan(strategy="holdout", seed=seed)
    try:
        indices = make_splits(target, plan)
    except ValueError as e:
        raise BaselineError(f"The table cannot be split: {e}") from e
    try:
        encoder = TargetEncoder.fit(target.iloc[indices.train], positive_class)
        y = encoder.transform(target)
    except ValueError as e:
        raise BaselineError(str(e)) from e
    if not encoder.is_binary:
        raise BaselineError(
            f"The run loop handles two-class targets; {target_column!r} has "
            f"{len(encoder.classes)} classes"
        )
    for part, rows in (
        ("training", indices.train),
        ("validation", indices.val),
        ("test", indices.test),
    ):
        if len(set(y[rows])) < 2:
            raise BaselineError(f"the {part} rows have only one class; there is nothing to learn")

    pool = np.sort(np.concatenate([indices.train, indices.val]))
    spec = {**DEFAULT_RULE, **(rule or {})}
    splitter = RepeatedStratifiedKFold(
        n_splits=spec["n_splits"], n_repeats=spec["n_repeats"], random_state=seed
    )
    folds = [TableFold(pool[a], pool[b]) for a, b in splitter.split(pool, y[pool])]
    split = TableSplit(plan, indices.train, indices.val, indices.test, folds, pool)

    frame = model_frame(features)
    try:
        model, metrics, shares = fit_validated(frame, y, split.train, split.val, seed)
    except SplitError as e:
        raise BaselineError(str(e)) from e
    ranked = sorted(shares.items(), key=lambda p: -p[1])
    kept = [
        KeptFeature(
            Candidate(name, "attribute", 0, f"the column {name} of the table", ""), float(share)
        )
        for name, share in ranked
    ]
    result = BaselineResult(
        features=kept,
        dropped=[],
        skipped_tables=[],
        candidates=frame.shape[1],
        metrics=metrics,
        split=split.summary(),
        params={**LGBM_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        seed=seed,
        engine="lightgbm",
        engine_version=lgb.__version__,
        seconds=time.monotonic() - started,
        model=model,
        frame=frame,
        labels=pd.DataFrame({"label": y}),
        train=split.train,
        graph=None,
        temporal=split,
        fold_kind="random",
        report_threshold=True,
    )
    return result, TableTask(target_column, features, y, encoder, report, split)
