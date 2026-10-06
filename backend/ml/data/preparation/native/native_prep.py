"""Native sklearn-based DataPreparationProvider."""

from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.core.interfaces import (
    DataPreparationProvider,
    LeakageWarning,
    ProfileResult,
    ReadinessReport,
)
from ml.data.profiling.profiler import DataProfiler
from ml.validation.leakage import scan


class NativeDataPreparationProvider(DataPreparationProvider):
    """Native sklearn-based data preparation provider."""

    def __init__(self) -> None:
        self._profiler = DataProfiler()

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """Profile the DataFrame using DataProfiler."""
        return self._profiler.profile(df)

    def assess_readiness(self, df: pd.DataFrame, target_column: str) -> ReadinessReport:
        """Assess data readiness using profiling + leakage detection."""
        profile = self._profiler.profile(df, target_column=target_column)
        leakage_warnings = self.detect_leakage(df, target_column)

        # Derive risk scores from profile
        flagged = [w for w in leakage_warnings if w.flagged]
        leakage_risk = "none"
        if flagged:
            leakage_risk = "high" if any(w.severity == "block" for w in flagged) else "medium"

        validation_risk = "none"
        if profile.rows < 1000:
            validation_risk = "high"
        elif profile.rows < 5000:
            validation_risk = "medium"

        feature_risk = "none"
        high_missing_cols = [
            col for col in df.columns if col != target_column and df[col].isna().mean() > 0.3
        ]
        if len(high_missing_cols) > 5:
            feature_risk = "high"
        elif len(high_missing_cols) > 0:
            feature_risk = "medium"

        # Data quality score (0-1)
        dq_score = max(0.0, 1.0 - profile.missing_rate - (0.1 if flagged else 0.0))

        recommendations: list[str] = []
        if validation_risk == "high":
            recommendations.append("Dataset is small (<1000 rows). Use stratified k-fold CV.")
        if feature_risk != "none":
            recommendations.append(f"Columns with high missingness: {high_missing_cols}")
        recommendations.extend(profile.warnings)

        return ReadinessReport(
            data_quality_score=dq_score,
            leakage_risk=leakage_risk,
            validation_risk=validation_risk,
            feature_risk=feature_risk,
            warnings=leakage_warnings,
            recommendations=recommendations,
        )

    def detect_leakage(self, df: pd.DataFrame, target_column: str) -> list[LeakageWarning]:
        """Run every leakage check (ml/validation/leakage.py)."""
        return scan(df, target_column)

    def prepare(self, df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
        """Transform df according to config using sklearn Pipeline.

        Args:
            df: DataFrame to prepare.
            config: Prep config containing 'target_column', 'numeric_imputer', etc.

        Returns:
            Tuple of (transformed_df, fitted_sklearn_pipeline)
        """
        target_column = config.get("target_column")
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' missing for preparation.")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Find columns
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object", "category"]).columns

        # Build pipeline
        num_imputer_strategy = config.get("numeric_imputer", "median")
        cat_imputer_strategy = config.get("categorical_imputer", "constant")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy=num_imputer_strategy)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy=cat_imputer_strategy, fill_value="missing"),
                            ),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        transformed_X = preprocessor.fit_transform(X)

        # Reconstruct DataFrame (for downstream previewing/debugging)
        num_cols = list(numeric_features)

        # One-hot column names; a failure here is a bug and is raised, not papered over with made-up names.
        cat_cols = []
        if len(categorical_features) > 0:
            cat_cols = list(
                preprocessor.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_features)
            )

        all_cols = num_cols + cat_cols

        if transformed_X.shape[1] == len(all_cols):
            transformed_df = pd.DataFrame(transformed_X, columns=all_cols, index=df.index)
        else:
            # Fallback if dimension mismatch
            transformed_df = pd.DataFrame(transformed_X, index=df.index)

        transformed_df[target_column] = y

        return transformed_df, preprocessor

    def export_pipeline(self, pipeline: Any, path: str) -> str:
        """Serialize pipeline to path using joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(pipeline, path)
        return path
