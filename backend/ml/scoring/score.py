"""Score the entities eligible at one cutoff with a run's champion (#62).

The queries are the ones the run trained with: ``entities.sql`` is the ``eligible`` step of the
task's label query for one cutoff, each feature is its stored SQL with ``__labels`` defined from
those entities. Pure functions; the service runs them through the SQL guard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sqlglot

from ml.data.schema_graph import SchemaGraph
from ml.export.bundle import CUTOFF_TOKEN, entity_query, transpile
from ml.tasks.spec import TaskSpec

REASONS = 3
NULL_RATE_TOLERANCE = 0.10  # absolute difference in the share of missing values
MEAN_TOLERANCE_STDS = 3.0
COUNT_TOLERANCE = 0.5  # relative difference in eligible entities per cutoff


class SchemaMismatch(ValueError):
    """The database no longer has what the champion's features read."""


@dataclass
class Check:
    name: str
    detail: str


@dataclass
class Scored:
    rows: pd.DataFrame  # entity_id, score, rank, decile, reason_1..3
    features: pd.DataFrame
    warnings: list[Check] = field(default_factory=list)


def stamp(cutoff: datetime) -> str:
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


def entities_sql(spec: TaskSpec, graph: SchemaGraph, dialect: str, cutoff: datetime) -> str:
    return entity_query(spec, graph, dialect).replace(CUTOFF_TOKEN, stamp(cutoff))


def feature_sql(entities: str, feature: str, dialect: str) -> str:
    """The stored feature query with ``__labels`` defined from the entity query."""
    sql = transpile(feature, dialect)
    ctes = "__labels AS (SELECT entity_id, cutoff_time FROM eligible)"
    match = re.match(r"\s*WITH\s+(RECURSIVE\s+)?", sql, flags=re.IGNORECASE)
    tree = sqlglot.parse_one(entities, read=dialect)
    with_clause = tree.args.get("with_") or tree.args["with"]  # the key differs between versions
    declared = ", ".join(c.sql(dialect=dialect) for c in with_clause.expressions)
    if match:
        return f"WITH {match.group(1) or ''}{declared}, {ctes}, {sql[match.end() :]}"
    return f"WITH {declared}, {ctes}\n{sql}"


def required_columns_missing(
    trained: dict[str, list[str]], graph: SchemaGraph, features: dict[str, str]
) -> list[str]:
    """Messages for every column the training data had and the database no longer has."""
    now = {t.key: {c.name for c in t.columns} for t in graph.tables}
    now.update({t.name: {c.name for c in t.columns} for t in graph.tables})
    problems = []
    for table, columns in sorted(trained.items()):
        have = now.get(table)
        if have is None:
            problems.append(f"table {table} is missing")
            continue
        for column in columns:
            if column in have:
                continue
            users = sorted(
                name
                for name, sql in features.items()
                if re.search(rf"\b{re.escape(column)}\b", sql)
            )
            used = f" (used by {', '.join(users[:3])})" if users else ""
            problems.append(f"column {table}.{column} is missing{used}")
    return problems


def predict_with_reasons(
    booster: lgb.Booster, frame: pd.DataFrame, k: int = REASONS
) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
    names = booster.feature_name()
    x = frame[names]
    score = np.asarray(booster.predict(x))
    contrib = np.asarray(booster.predict(x, pred_contrib=True))[:, :-1]
    order = np.argsort(-np.abs(contrib), axis=1)[:, :k]
    reasons = [
        [(names[i], float(row[i])) for i in idx] for row, idx in zip(contrib, order, strict=True)
    ]
    return score, reasons


def rank_rows(
    entity_ids: pd.Series, score: np.ndarray, reasons: list[list[tuple[str, float]]]
) -> pd.DataFrame:
    frame = pd.DataFrame({"entity_id": entity_ids.to_numpy(), "score": score})
    order = np.argsort(-score, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["decile"] = np.minimum(10, 1 + (frame["rank"] - 1) * 10 // max(len(frame), 1))
    for i in range(REASONS):
        frame[f"reason_{i + 1}"] = [
            f"{r[i][0]} ({'+' if r[i][1] >= 0 else '-'}{abs(r[i][1]):.3f})" if len(r) > i else ""
            for r in (reasons[j] for j in order)
        ]
    return frame


def drift_warnings(
    features: pd.DataFrame, trained: dict[str, dict[str, float]], n_scored: int, per_cutoff: float
) -> list[Check]:
    """Compare what is being scored with what the model was trained on. Warn, never block."""
    out: list[Check] = []
    for name, stats in trained.items():
        if name not in features.columns or stats.get("mean") is None:
            continue
        col = pd.to_numeric(features[name], errors="coerce")
        gap = abs(float(col.isna().mean()) - float(stats["null_rate"]))
        if gap > NULL_RATE_TOLERANCE:
            out.append(
                Check(
                    "null_rate",
                    f"{name}: {col.isna().mean():.0%} missing now, {stats['null_rate']:.0%} in training",
                )
            )
        std = float(stats.get("std") or 0.0)
        if std > 0 and col.notna().any():
            shift = abs(float(col.mean()) - float(stats["mean"])) / std
            if shift > MEAN_TOLERANCE_STDS:
                out.append(
                    Check(
                        "mean_shift",
                        f"{name}: mean {col.mean():.4g} now, {stats['mean']:.4g} in training "
                        f"({shift:.1f} standard deviations)",
                    )
                )
    if per_cutoff > 0 and abs(n_scored - per_cutoff) / per_cutoff > COUNT_TOLERANCE:
        out.append(
            Check(
                "entity_count",
                f"{n_scored} eligible entities now, {per_cutoff:.0f} per cutoff in training",
            )
        )
    return out


def feature_stats(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, dict[str, float]]:
    """Null rate, mean and standard deviation of every numeric feature on the training rows."""
    out: dict[str, dict[str, float]] = {}
    for name in frame.columns:
        col = frame[name].iloc[rows]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        out[str(name)] = {
            "null_rate": float(col.isna().mean()),
            "mean": float(col.mean()) if col.notna().any() else 0.0,
            "std": float(col.std()) if col.notna().sum() > 1 else 0.0,
        }
    return out


def summary(rows: pd.DataFrame, top_k: int) -> dict[str, Any]:
    top = rows.head(top_k)
    return {
        "n_scored": len(rows),
        "top_k": len(top),
        "score_at_k": float(top["score"].min()) if len(top) else None,
        # the sum of the model's probabilities, which are not calibrated (no calibrator is fitted)
        "expected_positives_in_top_k_uncalibrated": float(top["score"].sum()),
        "expected_positives_overall_uncalibrated": float(rows["score"].sum()),
    }
