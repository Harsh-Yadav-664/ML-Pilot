"""MetricProvider ABC (re-export)."""
from ml.core.interfaces import MetricProvider, ClassificationMetrics, RegressionMetrics

__all__ = ["MetricProvider", "ClassificationMetrics", "RegressionMetrics"]
