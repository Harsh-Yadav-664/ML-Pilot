"""Native sklearn-based DataPreparationProvider."""
from __future__ import annotations

import os
import joblib
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.core.interfaces import (
    DataPreparationProvider,
    ProfileResult,
    ReadinessReport,
    LeakageWarning,
)
from ml.data.profiling.profiler import DataProfiler
from ml.validation.leakage import LeakageDetector


class NativeDataPreparationProvider(DataPreparationProvider):
    """Native sklearn-based data preparation provider."""

    def __init__(self) -> None:
        self._profiler = DataProfiler()
        self._leakage_detector = LeakageDetector()

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """Profile the DataFrame using DataProfiler."""
        return self._profiler.profile(df)

    def assess_readiness(self, df: pd.DataFrame, target_column: str) -> ReadinessReport:
        """Assess data readiness using profiling + leakage detection."""
        profile = self._profiler.profile(df, target_column=target_column)
        leakage_warnings = self.detect_leakage(df, target_column)

        # Derive risk scores from profile
        leakage_risk = "none"
        if leakage_warnings:
            high_severity = [w for w in leakage_warnings if w.severity == "high"]
            leakage_risk = "high" if high_severity else "medium"

        validation_risk = "none"
        if profile.rows < 1000:
            validation_risk = "high"
        elif profile.rows < 5000:
            validation_risk = "medium"

        feature_risk = "none"
        high_missing_cols = [
            col for col in df.columns
            if col != target_column and df[col].isna().mean() > 0.3
        ]
        if len(high_missing_cols) > 5:
            feature_risk = "high"
        elif len(high_missing_cols) > 0:
            feature_risk = "medium"

        # Data quality score (0-1)
        dq_score = max(0.0, 1.0 - profile.missing_rate - (0.1 if leakage_warnings else 0.0))

        recommendations: list[str] = []
        if validation_risk == "high":
            recommendations.append("Dataset is small (<1000 rows). Use stratified k-fold CV.")
        if feature_risk != "none":
            recommendations.append(f"Columns with high missingness: {high_missing_cols}")
        for w in profile.warnings:
            recommendations.append(w)

        return ReadinessReport(
            data_quality_score=dq_score,
            leakage_risk=leakage_risk,
            validation_risk=validation_risk,
            feature_risk=feature_risk,
            warnings=leakage_warnings,
            recommendations=recommendations,
        )

    def detect_leakage(self, df: pd.DataFrame, target_column: str) -> list[LeakageWarning]:
        """Run all leakage detectors."""
        warnings: list[LeakageWarning] = []
        warnings.extend(self._leakage_detector.detect_target_leakage(df, target_column))
        warnings.extend(self._leakage_detector.detect_timestamp_leakage(df, target_column))
        warnings.extend(self._leakage_detector.detect_entity_leakage(df, target_column))
        warnings.extend(self._leakage_detector.detect_preprocessing_leakage(df, target_column))
        return warnings

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
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Build pipeline
        num_imputer_strategy = config.get("numeric_imputer", "median")
        cat_imputer_strategy = config.get("categorical_imputer", "constant")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=cat_imputer_strategy, fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ])

        transformed_X = preprocessor.fit_transform(X)
        
        # Reconstruct DataFrame (for downstream previewing/debugging)
        num_cols = list(numeric_features)
        
        # Safely handle one-hot encoder feature names
        cat_cols = []
        if len(categorical_features) > 0:
            try:
                # get_feature_names_out might fail if the OneHotEncoder has issues, fallback safely
                cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
            except Exception:
                cat_cols = [f"cat_{i}" for i in range(transformed_X.shape[1] - len(num_cols))]

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
