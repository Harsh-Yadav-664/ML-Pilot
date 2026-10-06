"""Deterministic clean-up of the feature frame before training.

Two problems make naive one-hot pipelines slow and wrong on real tables:
ID columns (one category per row) and numbers stored as text (e.g. blanks
in a numeric column make pandas read it as strings). Both are handled here,
and everything that was changed is reported so it can be recorded on the run.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# One-hot encode at most this many categories per column; rarer ones are grouped.
ONEHOT_MAX_CATEGORIES = 20

# Share of non-empty values that must parse as numbers before a text column is converted.
NUMERIC_TEXT_MIN_SHARE = 0.95


def _looks_like_id_name(name: str) -> bool:
    """True for "id", "user_id", "customerID", "orderId"."""
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id") or bool(re.search(r"[a-z](ID|Id)$", name))


def find_id_columns(X: pd.DataFrame) -> dict[str, str]:
    """Return {column: reason} for columns that identify rows rather than describe them."""
    n_rows = len(X)
    found: dict[str, str] = {}
    if n_rows < 2:
        return found
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_float_dtype(series):
            continue
        n_unique = series.nunique(dropna=True)
        if n_unique == n_rows:
            found[col] = "id-like: one unique value per row"
        elif _looks_like_id_name(str(col)) and n_unique >= 0.9 * n_rows:
            found[col] = "id-like: name ends in id and values are almost all unique"
    return found


def coerce_numeric_text(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert text columns that are mostly numbers to numeric.

    Returns the new frame and {column: number of non-null cells that failed to
    parse}; those cells become NaN (and are imputed later), and the count is
    reported instead of being hidden.
    """
    X = X.copy()
    failures: dict[str, int] = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        raw = X[col]
        present = raw.notna()
        n_present = int(present.sum())
        if n_present == 0:
            continue
        parsed = pd.to_numeric(raw.astype("string").str.strip(), errors="coerce")
        n_ok = int(parsed[present].notna().sum())
        if n_ok == 0 or n_ok / n_present < NUMERIC_TEXT_MIN_SHARE:
            continue
        X[col] = parsed.astype("float64")
        failures[col] = n_present - n_ok
    return X, failures


def prepare_feature_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop ID columns and convert numeric text. Returns (frame, report)."""
    X, numeric_failures = coerce_numeric_text(X)
    excluded = find_id_columns(X)
    X = X.drop(columns=list(excluded))
    report = {
        "excluded_features": excluded,
        "numeric_coercion": numeric_failures,
    }
    return X, report
