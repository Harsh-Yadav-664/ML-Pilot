"""LeakageDetector: detect common forms of data leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.core.interfaces import LeakageWarning


class LeakageDetector:
    """Detect data leakage patterns in a DataFrame."""

    TIMESTAMP_KEYWORDS = ("date", "time", "timestamp", "created", "updated", "modified", "dt", "ts")
    ID_KEYWORDS = ("id", "_id", "uid", "uuid", "key", "user_id", "customer_id")

    def detect_target_leakage(
        self, df: pd.DataFrame, target_column: str
    ) -> list[LeakageWarning]:
        """Detect columns that are likely derived from or identical to the target.

        Checks:
        - Pearson correlation > 0.95 (numeric)
        - Columns containing the target name as a substring
        """
        warnings: list[LeakageWarning] = []
        if target_column not in df.columns:
            return warnings

        target = df[target_column]

        for col in df.columns:
            if col == target_column:
                continue

            # Name-based heuristic
            if target_column.lower() in col.lower() and col != target_column:
                warnings.append(
                    LeakageWarning(
                        column=col,
                        leakage_type="target",
                        severity="high",
                        reason=f"Column name '{col}' contains target name '{target_column}' — likely derived.",
                        suggested_action=f"Remove or carefully audit column '{col}' before training.",
                    )
                )
                continue

            # Correlation-based check
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
                try:
                    valid = df[[col, target_column]].dropna()
                    if len(valid) > 10:
                        corr = abs(float(valid[col].corr(valid[target_column])))
                        if corr > 0.95:
                            warnings.append(
                                LeakageWarning(
                                    column=col,
                                    leakage_type="target",
                                    severity="high",
                                    reason=f"Column '{col}' has correlation {corr:.3f} with target — possible leakage.",
                                    suggested_action=f"Investigate whether '{col}' is computed after target is known.",
                                )
                            )
                except Exception:
                    pass

        return warnings

    def detect_timestamp_leakage(
        self, df: pd.DataFrame, target_column: str
    ) -> list[LeakageWarning]:
        """Detect timestamp/date columns that may cause temporal leakage."""
        warnings: list[LeakageWarning] = []
        for col in df.columns:
            if col == target_column:
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in self.TIMESTAMP_KEYWORDS):
                warnings.append(
                    LeakageWarning(
                        column=col,
                        leakage_type="temporal",
                        severity="medium",
                        reason=f"Column '{col}' appears to be a timestamp — may cause temporal leakage if not handled.",
                        suggested_action="Use TimeSeriesSplit validation and drop or encode timestamps carefully.",
                    )
                )
        return warnings

    def detect_entity_leakage(
        self, df: pd.DataFrame, target_column: str
    ) -> list[LeakageWarning]:
        """Detect entity ID columns that may cause group leakage."""
        warnings: list[LeakageWarning] = []
        for col in df.columns:
            if col == target_column:
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in self.ID_KEYWORDS):
                warnings.append(
                    LeakageWarning(
                        column=col,
                        leakage_type="entity",
                        severity="low",
                        reason=f"Column '{col}' appears to be an entity ID — may cause group leakage.",
                        suggested_action="Use GroupKFold or remove ID columns from features.",
                    )
                )
        return warnings

    def detect_preprocessing_leakage(
        self, df: pd.DataFrame, target_column: str
    ) -> list[LeakageWarning]:
        """Detect columns that suggest preprocessing was done on the full dataset.

        This is a heuristic — looks for columns with suspiciously low variance
        or perfect normalization that suggest global fitting.
        """
        warnings: list[LeakageWarning] = []
        for col in df.columns:
            if col == target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) > 0:
                    col_min = float(series.min())
                    col_max = float(series.max())
                    if abs(col_min) < 1e-9 and abs(col_max - 1.0) < 1e-6:
                        warnings.append(
                            LeakageWarning(
                                column=col,
                                leakage_type="preprocessing",
                                severity="medium",
                                reason=f"Column '{col}' appears to be [0,1]-normalized — suspect global preprocessing.",
                                suggested_action="Ensure normalization is fit only on training data.",
                            )
                        )
        return warnings
