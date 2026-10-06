"""Ranking metrics for binary tasks: how good is the top of the list a business acts on.

A retention team calls the top k customers, so what matters is how many of those k are
real positives (precision@k), how many of all positives they reach (recall@k), and how
much better that is than picking at random (lift@k = precision@k / base rate).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score

# Default cut-offs: the top 1%, 5% and 10% of rows.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
# And an absolute list size ("call the top 100 customers"), when there are that many rows.
DEFAULT_ABSOLUTE_K: tuple[int, ...] = (100,)


def _top_k(y_true: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Labels of the k highest scores. Ties keep row order (stable), so results repeat."""
    if not 1 <= k <= len(y_true):
        raise ValueError(f"k must be between 1 and {len(y_true)}, got {k}")
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return np.asarray(y_true)[order[:k]]


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    return float(_top_k(y_true, score, k).mean())


def recall_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    positives = int(np.sum(y_true))
    if positives == 0:
        raise ValueError("recall@k is undefined without positives")
    return float(_top_k(y_true, score, k).sum() / positives)


def lift_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    rate = base_rate(y_true)
    if rate == 0:
        raise ValueError("lift@k is undefined without positives")
    return precision_at_k(y_true, score, k) / rate


def base_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    return float(average_precision_score(y_true, score))


def k_for_fraction(n: int, fraction: float) -> int:
    """Rows in the top `fraction` of n: rounded up, at least 1."""
    return max(1, math.ceil(n * fraction - 1e-9))


def fraction_key(fraction: float) -> str:
    """0.1 -> '10pct', 0.005 -> '0_5pct' (metric names stay plain identifiers)."""
    pct = round(fraction * 100, 4)
    return (f"{pct:g}".replace(".", "_")) + "pct"


def ranking_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    absolute_k: Sequence[int] = (),
) -> dict[str, float]:
    """Base rate, PR-AUC, and precision/recall/lift at each cut-off.

    Requires both classes in y_true (callers record the metric as unavailable otherwise).
    The trivial baseline (score every row with the base rate) has PR-AUC = base rate and
    lift 1 at every k, so it is reported as `trivial_pr_auc` for comparison.
    """
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    rate = base_rate(y)
    out: dict[str, float] = {"base_rate": rate, "pr_auc": pr_auc(y, s), "trivial_pr_auc": rate}
    cuts = [(fraction_key(f), k_for_fraction(len(y), f)) for f in fractions]
    cuts += [(str(k), k) for k in absolute_k if k <= len(y)]
    for name, k in cuts:
        out[f"precision_at_{name}"] = precision_at_k(y, s, k)
        out[f"recall_at_{name}"] = recall_at_k(y, s, k)
        out[f"lift_at_{name}"] = lift_at_k(y, s, k)
    return out
