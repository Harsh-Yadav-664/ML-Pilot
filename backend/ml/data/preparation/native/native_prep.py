"""Native sklearn-based DataPreparationProvider.

Phase 1 placeholder — methods raise NotImplementedError with TODO comments.
Full implementation will use sklearn Pipeline + ColumnTransformer.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ml.core.interfaces import (
    DataPreparationProvider,
    ProfileResult,
    ReadinessReport,
    LeakageWarning,
)
from ml.data.profiling.profiler import DataProfiler
from ml.validation.leakage import LeakageDetector


class NativeDataPreparationProvider(DataPreparationProvider):
    """Native sklearn-based data preparation provider.

    profile() and assess_readiness() are functional.
    prepare() and export_pipeline() are Phase 1 stubs.
    """

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
        return warnings

    def prepare(self, df: pd.DataFrame, config: dict[str, Any]) -> tuple[Any, Any]:
        """Transform df according to config.

        TODO (Phase 1): Implement sklearn Pipeline + ColumnTransformer.
        Will handle: imputation, encoding, scaling, feature selection.
        """
        raise NotImplementedError(
            "NativeDataPreparationProvider.prepare() is a Phase 1 feature. "
            "Use a DataCleanAdapter or implement the full pipeline in Phase 1."
        )

    def export_pipeline(self, pipeline: Any, path: str) -> str:
        """Serialize pipeline to path.

        TODO (Phase 1): Use joblib.dump() to serialize the sklearn Pipeline.
        """
        raise NotImplementedError(
            "NativeDataPreparationProvider.export_pipeline() is a Phase 1 feature."
        )
