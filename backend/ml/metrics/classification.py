"""Classification metrics computation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.core.errors import describe, unavailable
from ml.metrics.calibration import Calibrator, brier_score, expected_calibration_error
from ml.metrics.ranking import DEFAULT_ABSOLUTE_K, DEFAULT_FRACTIONS, ranking_metrics


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
    average = "binary" if binary else "weighted"
    metrics: dict[str, float | None] = {
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


def binary_business_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    calibrator: Calibrator,
    threshold: float,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    absolute_k: Sequence[int] = DEFAULT_ABSOLUTE_K,
    notes: dict[str, str] | None = None,
) -> dict[str, float | None]:
    """What a business user reads for a binary task: base rate, PR-AUC, precision /
    recall / lift at the top k, calibration (Brier, ECE) and the results at the chosen
    threshold. `score` is the raw positive-class probability; the calibrator and the
    threshold were fitted on validation rows and are applied here unchanged.
    """
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    metrics: dict[str, float | None] = {}
    if len(np.unique(y)) < 2:
        for name in ("base_rate", "pr_auc"):
            unavailable(metrics, notes, name, "only one class present")
    else:
        # Calibration is monotone, so ranking on the raw score gives the same order.
        metrics.update(ranking_metrics(y, s, fractions, absolute_k))
    prob = calibrator.apply(s)
    metrics["brier"] = brier_score(y, prob)
    metrics["ece"] = expected_calibration_error(y, prob)
    pred = (prob >= threshold).astype(int)
    metrics["precision_at_threshold"] = float(precision_score(y, pred, zero_division=0))
    metrics["recall_at_threshold"] = float(recall_score(y, pred, zero_division=0))
    metrics["f1_at_threshold"] = float(f1_score(y, pred, zero_division=0))
    return metrics
