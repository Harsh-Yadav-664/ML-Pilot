"""Classification metrics computation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.core.errors import describe, unavailable

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    notes: dict[str, str] | None = None,
) -> dict[str, float | None]:
    """Compute standard classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities (optional).
        notes: Filled with the reason for every metric recorded as None.

    Returns:
        Dictionary of metric name to value; None (with a note) when a metric is undefined.
    """
    # Labels are encoded by ml.core.targets.TargetEncoder: for binary targets
    # 1 is the positive class, so precision/recall/F1 are reported for it.
    binary = set(np.unique(y_true)) | set(np.unique(y_pred)) <= {0, 1}
    average = 'binary' if binary else 'weighted'
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # ROC-AUC for binary targets when probabilities are available.
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            unavailable(metrics, notes, "roc_auc", "only one class present")
        elif n_classes == 2:
            try:
                # y_prob has shape (n_samples, n_classes); class 1 is the positive class
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except ValueError as e:
                unavailable(metrics, notes, "roc_auc", f"could not be computed: {describe(e)}")

    return metrics
