"""Shared classification target encoding.

Labels are fitted on training labels only. For binary targets the positive
class is encoded as 1 (default: the minority class in the training labels;
ties go to the last label in sorted order), so binary metrics and
`predict_proba[:, 1]` refer to it. Multiclass targets use sorted order, as
sklearn's LabelEncoder does.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _native(value: Any) -> Any:
    """Convert numpy scalars to plain Python values so the mapping is JSON-safe."""
    return value.item() if isinstance(value, np.generic) else value


class TargetEncoder:
    """Maps class labels to 0..k-1 and back."""

    def __init__(self, classes: list[Any], positive_class: Any | None = None) -> None:
        if len(classes) < 2:
            raise ValueError(f"Classification target needs at least 2 classes, got {classes}")
        if len(set(classes)) != len(classes):
            raise ValueError(f"Duplicate classes in target encoding: {classes}")
        self.classes = list(classes)
        self.positive_class = positive_class
        self._index = {c: i for i, c in enumerate(self.classes)}

    @classmethod
    def fit(cls, y_train: Iterable[Any], positive_class: Any | None = None) -> TargetEncoder:
        counts = pd.Series(list(y_train)).dropna().map(_native).value_counts()
        if counts.empty:
            raise ValueError("Target column has no non-missing values")
        labels = sorted(counts.index, key=lambda v: (str(type(v)), v))
        if len(labels) == 2:
            if positive_class is None:
                # Minority class; on a tie, the last label in sorted order.
                positive_class = min(reversed(labels), key=lambda v: counts[v])
            elif positive_class not in labels:
                raise ValueError(
                    f"positive_class {positive_class!r} is not one of the target labels {labels}"
                )
            negative = next(v for v in labels if v != positive_class)
            return cls([negative, positive_class], positive_class)
        return cls(labels, None)

    @property
    def is_binary(self) -> bool:
        return len(self.classes) == 2

    def transform(self, y: Iterable[Any]) -> np.ndarray:
        values = [_native(v) for v in y]
        unknown = sorted({str(v) for v in values if v not in self._index})
        if unknown:
            raise ValueError(f"Target labels not seen in training data: {unknown}")
        return np.array([self._index[v] for v in values], dtype=int)

    def inverse_transform(self, codes: Iterable[Any]) -> list[Any]:
        return [self.classes[int(c)] for c in codes]

    def to_dict(self) -> dict[str, Any]:
        return {"classes": self.classes, "positive_class": self.positive_class}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetEncoder:
        return cls(list(data["classes"]), data.get("positive_class"))
