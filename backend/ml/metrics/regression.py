"""Regression metric computation."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ml.core.interfaces import MetricProvider, ClassificationMetrics, RegressionMetrics


class SklearnRegressionMetricProvider(MetricProvider):
    """Compute regression metrics using scikit-learn."""

    def compute_regression(self, y_true: Any, y_pred: Any) -> RegressionMetrics:
        from sklearn import metrics as skm

        rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
        mae = float(skm.mean_absolute_error(y_true, y_pred))
        r2 = float(skm.r2_score(y_true, y_pred))

        mape = float(skm.mean_absolute_percentage_error(y_true, y_pred))

        return RegressionMetrics(rmse=rmse, mae=mae, r2=r2, mape=mape)

    def compute_classification(self, y_true: Any, y_pred: Any, y_prob: Optional[Any] = None) -> ClassificationMetrics:
        raise NotImplementedError("Use SklearnClassificationMetricProvider for classification metrics.")

    def compare(self, baseline: dict[str, float], challenger: dict[str, float], primary_metric: str) -> dict[str, Any]:
        delta = challenger.get(primary_metric, 0) - baseline.get(primary_metric, 0)
        return {
            "primary_metric": primary_metric,
            "baseline": baseline.get(primary_metric),
            "challenger": challenger.get(primary_metric),
            "delta": delta,
            "improved": delta > 0 if primary_metric in ("r2",) else delta < 0,
            "all_deltas": {k: challenger.get(k, 0) - baseline.get(k, 0) for k in set(baseline) | set(challenger)},
        }
