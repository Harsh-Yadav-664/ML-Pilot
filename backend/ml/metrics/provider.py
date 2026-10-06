"""MetricProvider ABC (re-export)."""

from ml.core.interfaces import ClassificationMetrics, MetricProvider, RegressionMetrics

__all__ = ["ClassificationMetrics", "MetricProvider", "RegressionMetrics"]
