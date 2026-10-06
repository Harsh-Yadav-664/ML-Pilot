"""Built-in feature importances of a fitted sklearn pipeline, per original column.

This is the model's own importance (tree impurity/gain, or |coefficient| for
linear models), not SHAP. Encoded columns (e.g. one-hot) are summed back to
the source column so only real column names are reported.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

IMPORTANCE_METHOD = "model built-in importance (tree impurity/gain or |coefficient|), not SHAP"


def _source_column(encoded_name: str, columns: list[str]) -> str | None:
    # ColumnTransformer names look like "<transformer>__<column>[_<category>]"
    name = encoded_name.split("__", 1)[-1]
    matches = [c for c in columns if name == c or name.startswith(f"{c}_")]
    return max(matches, key=len) if matches else None


def builtin_importances(
    pipeline: Pipeline, columns: list[str], top_n: int = 10
) -> dict[str, float]:
    """Return {original column: share of total importance}, largest first.

    Raises ValueError if the model or preprocessor doesn't expose importances.
    """
    model = pipeline.steps[-1][1]
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float)).sum(axis=0)
    else:
        raise ValueError(f"{type(model).__name__} has no built-in feature importances")

    names = list(pipeline[:-1].get_feature_names_out())
    if len(names) != len(raw):
        raise ValueError(f"{len(names)} encoded features but {len(raw)} importances")

    totals: dict[str, float] = {}
    for name, value in zip(names, raw):
        col = _source_column(str(name), columns)
        if col is not None:
            totals[col] = totals.get(col, 0.0) + float(value)
    total = sum(totals.values())
    if total <= 0:
        raise ValueError("All feature importances are zero")
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {col: round(value / total, 4) for col, value in ranked}
