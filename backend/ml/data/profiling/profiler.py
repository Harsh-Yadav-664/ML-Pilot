"""DataProfiler: deterministic dataset profiling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.core.interfaces import ProfileResult
from ml.data.profiling.stats import column_stats


class DataProfiler:
    """Profile a pandas DataFrame and return a ProfileResult."""

    def profile(self, df: pd.DataFrame, target_column: str | None = None) -> ProfileResult:
        """Compute full profile for *df*.

        Args:
            df: The DataFrame to profile.
            target_column: Optional target column for class balance computation.

        Returns:
            A ProfileResult dataclass.
        """
        rows, cols = df.shape
        total_cells = rows * cols
        missing_cells = int(df.isna().sum().sum())
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0.0
        duplicate_rows = int(df.duplicated().sum())

        # Per-column stats
        col_stats: dict[str, Any] = {}
        for col in df.columns:
            col_stats[col] = column_stats(df[col])

        # Target balance
        target_balance: dict[str, Any] = {}
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
                # Regression-style target
                target_balance = {
                    "type": "continuous",
                    "mean": float(target_series.mean()),
                    "std": float(target_series.std()),
                    "min": float(target_series.min()),
                    "max": float(target_series.max()),
                }
            else:
                vc = target_series.value_counts(normalize=True)
                target_balance = {
                    "type": "categorical",
                    "distribution": {str(k): float(v) for k, v in vc.items()},
                }

        warnings = self._generate_warnings(df, missing_rate, duplicate_rows, target_balance)

        return ProfileResult(
            rows=rows,
            columns=cols,
            missing_rate=missing_rate,
            duplicate_rows=duplicate_rows,
            target_balance=target_balance,
            column_stats=col_stats,
            warnings=warnings,
        )

    def _generate_warnings(
        self, df: pd.DataFrame, missing_rate: float, duplicate_rows: int, target_balance: dict
    ) -> list[str]:
        warnings: list[str] = []
        if missing_rate > 0.2:
            warnings.append(f"High missing rate: {missing_rate:.1%} of all cells are missing.")
        if duplicate_rows > 0:
            warnings.append(f"{duplicate_rows} duplicate rows detected.")
        if target_balance.get("type") == "categorical" and target_balance.get("distribution"):
            dist = target_balance["distribution"]
            if dist:
                min_class_frac = min(dist.values())
                if min_class_frac < 0.05:
                    warnings.append(
                        f"Severe class imbalance detected: smallest class is {min_class_frac:.1%}."
                    )
        high_missing_cols = [col for col in df.columns if df[col].isna().mean() > 0.5]
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing: {high_missing_cols}")
        return warnings
