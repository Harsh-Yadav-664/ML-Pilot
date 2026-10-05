"""Classification metrics computation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    """Compute standard classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities (optional).
        
    Returns:
        Dictionary of metric name to float value.
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

    # Try computing ROC-AUC if probabilities are available and it's binary classification
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            # y_prob has shape (n_samples, n_classes); class 1 is the positive class
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            pass
            
    return metrics
