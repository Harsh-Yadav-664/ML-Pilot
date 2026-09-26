"""Column-level statistics helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def numeric_column_stats(series: pd.Series) -> dict[str, Any]:
    """Compute numeric statistics for a Series."""
    desc = series.describe()
    return {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean()),
        "mean": float(desc["mean"]) if "mean" in desc else None,
        "std": float(desc["std"]) if "std" in desc else None,
        "min": float(desc["min"]) if "min" in desc else None,
        "q25": float(desc["25%"]) if "25%" in desc else None,
        "median": float(desc["50%"]) if "50%" in desc else None,
        "q75": float(desc["75%"]) if "75%" in desc else None,
        "max": float(desc["max"]) if "max" in desc else None,
        "skew": float(series.skew()) if pd.api.types.is_numeric_dtype(series) else None,
        "kurtosis": float(series.kurtosis()) if pd.api.types.is_numeric_dtype(series) else None,
    }


def categorical_column_stats(series: pd.Series) -> dict[str, Any]:
    """Compute categorical statistics for a Series."""
    vc = series.value_counts()
    return {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean()),
        "n_unique": int(series.nunique()),
        "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
    }


def column_stats(series: pd.Series) -> dict[str, Any]:
    """Dispatch to numeric or categorical stat computation."""
    if pd.api.types.is_numeric_dtype(series):
        return numeric_column_stats(series)
    return categorical_column_stats(series)
