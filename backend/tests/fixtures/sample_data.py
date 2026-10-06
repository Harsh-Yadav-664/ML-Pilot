"""Sample DataFrames for testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_binary_classification_df(n_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Create a sample binary classification DataFrame."""
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 80, n_rows)
    income = rng.normal(50000, 15000, n_rows)
    score = rng.uniform(0, 1, n_rows)
    category = rng.choice(["A", "B", "C"], n_rows)
    target = (0.3 * (age / 80) + 0.4 * (income / 100000) + 0.3 * score > 0.5).astype(int)
    df = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "score": score.round(4),
            "category": category,
            "label": target,
        }
    )
    # Add some missing values
    df.loc[rng.choice(n_rows, 10, replace=False), "income"] = np.nan
    return df


def make_multiclass_classification_df(
    n_rows: int = 300, n_classes: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """Create a sample multiclass classification DataFrame."""
    rng = np.random.RandomState(random_state)
    x1 = rng.normal(0, 1, n_rows)
    x2 = rng.normal(0, 1, n_rows)
    x3 = rng.uniform(-2, 2, n_rows)
    target = np.digitize(x1 + x2 + x3, bins=[-1, 1]) % n_classes
    df = pd.DataFrame({"x1": x1.round(4), "x2": x2.round(4), "x3": x3.round(4), "class": target})
    return df


def make_regression_df(n_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Create a sample regression DataFrame."""
    rng = np.random.RandomState(random_state)
    x1 = rng.normal(0, 1, n_rows)
    x2 = rng.uniform(0, 10, n_rows)
    noise = rng.normal(0, 0.5, n_rows)
    target = 2 * x1 + 0.5 * x2 + noise
    df = pd.DataFrame({"x1": x1.round(4), "x2": x2.round(4), "price": target.round(4)})
    return df


def make_df_with_leakage(n_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Create a DataFrame that has target leakage and timestamp columns."""
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 80, n_rows)
    target = rng.randint(0, 2, n_rows)
    df = pd.DataFrame(
        {
            "age": age,
            "label": target,
            "label_copy": target,  # target leakage (same values)
            "user_id": range(n_rows),  # entity ID
            "created_date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),  # timestamp
        }
    )
    return df


def make_df_with_missing(
    n_rows: int = 100, missing_pct: float = 0.3, random_state: int = 42
) -> pd.DataFrame:
    """Create a DataFrame with high missing rate."""
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n_rows),
            "b": rng.normal(0, 1, n_rows),
            "c": rng.choice(["x", "y", "z"], n_rows),
            "target": rng.randint(0, 2, n_rows),
        }
    )
    # Add missing values
    n_missing = int(n_rows * missing_pct)
    df.loc[rng.choice(n_rows, n_missing, replace=False), "a"] = np.nan
    df.loc[rng.choice(n_rows, n_missing, replace=False), "b"] = np.nan
    return df


SAMPLE_BINARY_DF = make_binary_classification_df()
SAMPLE_MULTICLASS_DF = make_multiclass_classification_df()
SAMPLE_REGRESSION_DF = make_regression_df()
SAMPLE_LEAKAGE_DF = make_df_with_leakage()
SAMPLE_MISSING_DF = make_df_with_missing()
