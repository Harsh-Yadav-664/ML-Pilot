"""Probability calibration and the decision threshold, both fitted on validation rows only.

The test split never moves either: the executor fits them on validation predictions,
then applies them unchanged to the test predictions (AGENTS.md rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

N_BINS = 10
# Calibrate when the expected calibration error on validation is above this.
ECE_LIMIT = 0.05
# Isotonic regression needs enough rows not to overfit; below this use Platt scaling.
ISOTONIC_MIN_ROWS = 1000

CalibrationMethod = Literal["none", "isotonic", "platt"]


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Mean squared difference between the predicted probability and the 0/1 outcome."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def _bins(prob: np.ndarray, n_bins: int) -> np.ndarray:
    # Equal-width bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]; 1.0 falls in the last bin.
    return np.minimum((np.asarray(prob, dtype=float) * n_bins).astype(int), n_bins - 1)


def reliability_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Per probability bin: row count, mean predicted probability, observed positive rate."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    idx = _bins(p, n_bins)
    table = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        table.append(
            {
                "bin": b,
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": n,
                "mean_predicted": float(p[mask].mean()) if n else None,
                "observed_rate": float(y[mask].mean()) if n else None,
            }
        )
    return table


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """Row-weighted mean gap between predicted probability and observed rate per bin."""
    n = len(y_true)
    return float(
        sum(
            row["count"] / n * abs(row["mean_predicted"] - row["observed_rate"])
            for row in reliability_table(y_true, prob, n_bins)
            if row["count"]
        )
    )


@dataclass
class Calibrator:
    """Maps raw positive-class probabilities to calibrated ones."""

    method: CalibrationMethod
    model: IsotonicRegression | LogisticRegression | None = None

    def apply(self, prob: np.ndarray) -> np.ndarray:
        p = np.asarray(prob, dtype=float)
        if self.model is None:
            return p
        if isinstance(self.model, IsotonicRegression):
            return np.asarray(self.model.predict(p), dtype=float)
        return np.asarray(self.model.predict_proba(_logit(p))[:, 1], dtype=float)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)


def fit_calibrator(y_val: np.ndarray, prob_val: np.ndarray) -> tuple[Calibrator, dict]:
    """Calibrate only if validation ECE is above ECE_LIMIT; return it and what was done."""
    y, p = np.asarray(y_val), np.asarray(prob_val, dtype=float)
    before = expected_calibration_error(y, p)
    record: dict = {"ece_limit": ECE_LIMIT, "val_ece_before": before, "fitted_on": "validation"}
    if before <= ECE_LIMIT:
        return Calibrator("none"), {**record, "method": "none"}
    if len(y) >= ISOTONIC_MIN_ROWS:
        model: IsotonicRegression | LogisticRegression = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0
        ).fit(p, y)
        calibrator = Calibrator("isotonic", model)
    else:
        calibrator = Calibrator("platt", LogisticRegression().fit(_logit(p), y))
    # In-sample, so optimistic; the test ECE is the honest number.
    after = expected_calibration_error(y, calibrator.apply(p))
    return calibrator, {**record, "method": calibrator.method, "val_ece_after": after}


def choose_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> dict:
    """The probability cut-off with the highest F1 on validation rows (objective: max F1)."""
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)
    # The last precision/recall pair has no threshold.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    return {
        "objective": "max_f1",
        "value": float(thresholds[best]),
        "val_f1": float(f1[best]),
        "chosen_on": "validation",
    }
