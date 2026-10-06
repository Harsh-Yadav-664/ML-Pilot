"""Leakage checks for a single table, reported as evidence-carrying findings.

Each check is a small class with a `name`, a `category` from Kapoor & Narayanan's
taxonomy, and `run(table, target, context) -> list[Finding]`:

- L1  no train/test separation (e.g. duplicated rows that land on both sides)
- L2  illegitimate feature (target copies, post-outcome columns, features that
      alone predict the target almost perfectly)
- L3  test-set contamination via pre-processing (globally normalised columns)
- L4  temporal leakage (timestamp columns; point-in-time checks are #51/#54)
- ID  row identifiers (not leakage by themselves, but must not be features)

A finding with severity `block` or `warn` is a flag; `info` is context only.
The same Finding type is used by the evidence report (#60).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Severity = Literal["block", "warn", "info"]
FLAG_SEVERITIES = ("block", "warn")


@dataclass
class Finding:
    column: str | None
    check: str
    category: str
    severity: Severity
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.severity in FLAG_SEVERITIES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def name_tokens(name: str) -> list[str]:
    """Whole lower-case tokens of a column name: 'InternetService' -> ['internet', 'service']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def looks_like_datetime(series: pd.Series) -> bool:
    """True when (almost) all values of a column parse as dates or times."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().astype("string").head(200)
    if sample.empty or not sample.str.contains(r"\d", regex=True).all():
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.95)


def _is_binary_or_categorical(y: pd.Series) -> bool:
    return not pd.api.types.is_float_dtype(y) or y.nunique(dropna=True) <= 20


def _as_numeric(series: pd.Series) -> np.ndarray:
    """One column as a numeric vector for a small tree: categories become codes, NaN a sentinel."""
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(dtype="float64", copy=True)
    else:
        values = pd.factorize(series.astype("string"), sort=True)[0].astype("float64")
        values[series.isna().to_numpy()] = np.nan
    finite = values[np.isfinite(values)]
    sentinel = (finite.min() - 1.0) if finite.size else -1.0
    return np.where(np.isfinite(values), values, sentinel).reshape(-1, 1)


@dataclass
class Context:
    """Facts shared by the checks, computed once per scan."""

    id_columns: set[str] = field(default_factory=set)
    max_rows: int = 20000
    seed: int = 0


class Check:
    name = "check"
    category = "L2"

    def run(self, table: pd.DataFrame, target: str, context: Context) -> list[Finding]:
        raise NotImplementedError


class IdColumnCheck(Check):
    """Row identifiers: almost all values unique, and either text or monotonic integers."""

    name = "id_column"
    category = "ID"

    def run(self, table, target, context):
        out = []
        n = len(table)
        for col in table.columns.drop(target):
            s = table[col]
            if n < 20 or pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
                continue
            ratio = s.nunique(dropna=True) / n
            if ratio <= 0.95:
                continue
            if looks_like_datetime(s):
                continue  # unique timestamps are times, not identifiers (NameTokenCheck notes them)
            is_text = not pd.api.types.is_numeric_dtype(s)
            monotonic = (not is_text) and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
            if is_text or monotonic:
                context.id_columns.add(col)
                out.append(
                    Finding(
                        col,
                        self.name,
                        self.category,
                        "warn",
                        f"'{col}' identifies rows ({ratio:.0%} unique{', monotonic' if monotonic else ''}); "
                        "it should not be used as a feature.",
                        {
                            "unique_ratio": round(ratio, 4),
                            "monotonic": bool(monotonic),
                            "text": is_text,
                        },
                    )
                )
        return out


class TargetCopyCheck(Check):
    """A column that is the target under a one-to-one relabelling (e.g. 'Yes'/'No' -> 1/0)."""

    name = "target_copy"
    category = "L2"

    def run(self, table, target, context):
        out = []
        y = table[target]
        if not _is_binary_or_categorical(y):
            return self._continuous(table, target)
        for col in table.columns.drop(target):
            pair = table[[col, target]].dropna()
            if len(pair) < 10 or pair[col].nunique() < 2:
                continue
            forward = pair.groupby(col, observed=True)[target].nunique().max()
            backward = pair.groupby(target, observed=True)[col].nunique().max()
            if forward == 1 and backward == 1 and pair[col].nunique() == y.nunique(dropna=True):
                out.append(
                    Finding(
                        col,
                        self.name,
                        self.category,
                        "block",
                        f"'{col}' is a one-to-one relabelling of the target '{target}'.",
                        {"rows_checked": len(pair), "distinct_values": int(pair[col].nunique())},
                    )
                )
        return out

    def _continuous(self, table, target):
        """For a numeric target: a column that is a (near) perfectly monotone function of it."""
        out = []
        for col in table.columns.drop(target):
            if not pd.api.types.is_numeric_dtype(table[col]) or pd.api.types.is_bool_dtype(
                table[col]
            ):
                continue
            pair = table[[col, target]].dropna()
            if len(pair) < 10 or pair[col].nunique() < 3:
                continue
            rho = abs(float(pair[col].corr(pair[target], method="spearman")))
            if rho > 0.999:
                out.append(
                    Finding(
                        col,
                        self.name,
                        self.category,
                        "block",
                        f"'{col}' rises and falls with the target '{target}' almost exactly (rank correlation {rho:.4f}).",
                        {"spearman": round(rho, 6), "rows_checked": len(pair)},
                    )
                )
        return out


# Tokens that name something only known after the outcome (whole tokens only).
POST_OUTCOME_TOKENS = {
    "cancelled",
    "canceled",
    "cancellation",
    "churned",
    "refund",
    "refunded",
    "chargeback",
    "outcome",
    "result",
    "resolved",
    "resolution",
    "closed",
    "terminated",
    "termination",
    "default",
    "defaulted",
    "writeoff",
    "collections",
    "label",
    "after",
    "post",
    "final",
}
TIME_TOKENS = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "created",
    "updated",
    "modified",
    "at",
    "on",
}


class NameTokenCheck(Check):
    """Names that point at the target or at post-outcome information, matched as whole tokens."""

    name = "name_tokens"
    category = "L2"

    def run(self, table, target, context):
        out = []
        target_tokens = set(name_tokens(target)) - {"is", "has", "flag", "y"}
        for col in table.columns.drop(target):
            tokens = name_tokens(col)
            hits = sorted((set(tokens) & POST_OUTCOME_TOKENS) | (set(tokens) & target_tokens))
            if hits:
                out.append(
                    Finding(
                        col,
                        self.name,
                        self.category,
                        "warn",
                        f"'{col}' names {', '.join(repr(h) for h in hits)}, which suggests it is only known "
                        "after the outcome. Confirm it is available at prediction time.",
                        {"tokens": tokens, "matched": hits},
                    )
                )
            elif set(tokens) & {"date", "time", "timestamp", "datetime"} or (
                tokens[-1:] in (["at"], ["on"]) and len(tokens) > 1
            ):
                out.append(
                    Finding(
                        col,
                        self.name,
                        "L4",
                        "info",
                        f"'{col}' looks like a timestamp; use time-based splits if rows are ordered in time.",
                        {"tokens": tokens},
                    )
                )
        return out


class SingleFeatureCheck(Check):
    """One column alone predicts the target almost perfectly (3-fold CV, depth-3 tree)."""

    name = "single_feature_predictiveness"
    category = "L2"
    block_at = 0.98
    warn_at = 0.90

    def run(self, table, target, context):
        data = (
            table
            if len(table) <= context.max_rows
            else table.sample(context.max_rows, random_state=context.seed)
        )
        data = data[data[target].notna()]
        y_raw = data[target]
        classify = _is_binary_or_categorical(y_raw)
        if classify:
            y = pd.factorize(y_raw.astype("string"))[0]
            counts = np.bincount(y)
            if len(counts) < 2 or counts.min() < 3:
                return []
            folds = StratifiedKFold(3, shuffle=True, random_state=context.seed)
        else:
            y = y_raw.to_numpy(dtype="float64")
            folds = KFold(3, shuffle=True, random_state=context.seed)
        metric = "roc_auc" if classify else "r2"
        out = []
        for col in data.columns.drop(target):
            if col in context.id_columns and not pd.api.types.is_numeric_dtype(data[col]):
                continue  # text IDs are already reported; their codes are arbitrary
            X = _as_numeric(data[col])
            if np.unique(X).size < 2:
                continue
            score = self._cv_score(X, y, classify, folds, context.seed)
            if score is None or score <= self.warn_at:
                continue
            severity = "block" if score > self.block_at else "warn"
            out.append(
                Finding(
                    col,
                    self.name,
                    self.category,
                    severity,
                    f"'{col}' alone predicts '{target}' with {metric} {score:.3f}. Real signals rarely do; "
                    "check it is not derived from the outcome.",
                    {
                        "metric": metric,
                        "score": round(score, 4),
                        "folds": 3,
                        "model": "decision tree, depth 3",
                        "rows": len(data),
                    },
                )
            )
        return out

    @staticmethod
    def _cv_score(X, y, classify, folds, seed) -> float | None:
        scores = []
        for fit_idx, val_idx in folds.split(X, y):
            if classify:
                model = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(
                    X[fit_idx], y[fit_idx]
                )
                prob = model.predict_proba(X[val_idx])
                classes = model.classes_
                if len(np.unique(y[val_idx])) < 2:
                    continue
                if len(classes) == 2:
                    scores.append(roc_auc_score(y[val_idx], prob[:, 1]))
                else:
                    full = np.zeros((len(val_idx), int(y.max()) + 1))
                    full[:, classes] = prob
                    present = np.unique(y[val_idx])
                    full = full[:, present] / np.clip(
                        full[:, present].sum(axis=1, keepdims=True), 1e-12, None
                    )
                    scores.append(
                        roc_auc_score(y[val_idx], full, multi_class="ovr", labels=present)
                    )
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=seed).fit(
                    X[fit_idx], y[fit_idx]
                )
                scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))
        return float(np.mean(scores)) if scores else None


class DuplicateRowsCheck(Check):
    """Identical feature rows (IDs ignored): random splits put copies on both sides."""

    name = "duplicate_rows"
    category = "L1"
    warn_share = 0.05

    def run(self, table, target, context):
        features = table.drop(columns=[target, *context.id_columns])
        if features.shape[1] == 0 or len(features) < 20:
            return []
        dup_mask = features.duplicated(keep=False)
        share = float(dup_mask.mean())
        if share <= self.warn_share:
            return []
        conflicting = int(
            table[dup_mask]
            .groupby(list(features.columns), dropna=False, observed=True)[target]
            .nunique()
            .gt(1)
            .sum()
        )
        return [
            Finding(
                None,
                self.name,
                self.category,
                "warn",
                f"{share:.0%} of rows have an exact duplicate (ignoring IDs). A random split puts copies in both "
                "train and test, which inflates test scores.",
                {
                    "duplicated_rows": int(dup_mask.sum()),
                    "share": round(share, 4),
                    "duplicate_groups_with_different_targets": conflicting,
                },
            )
        ]


class GlobalNormalisationCheck(Check):
    """Continuous columns that look min-max scaled or z-scored on the whole file."""

    name = "global_normalisation"
    category = "L3"

    def run(self, table, target, context):
        out = []
        for col in table.columns.drop(target):
            s = table[col]
            if not pd.api.types.is_float_dtype(s):
                continue
            v = s.dropna()
            if len(v) < 20 or v.nunique() <= 2 or np.allclose(v, v.round()):
                continue
            lo, hi, mean, std = (
                float(v.min()),
                float(v.max()),
                float(v.mean()),
                float(v.std(ddof=0)),
            )
            minmax = abs(lo) < 1e-12 and abs(hi - 1.0) < 1e-9
            zscore = abs(mean) < 1e-9 and (
                abs(std - 1.0) < 1e-6 or abs(float(v.std(ddof=1)) - 1.0) < 1e-6
            )
            if minmax or zscore:
                kind = (
                    "min-max scaled to exactly [0, 1]"
                    if minmax
                    else "standardised to mean 0, std 1"
                )
                out.append(
                    Finding(
                        col,
                        self.name,
                        self.category,
                        "warn",
                        f"'{col}' is {kind} over the whole file, so test rows influenced the scaling. "
                        "Use the raw column; MLPilot fits scaling on training rows only.",
                        {"min": lo, "max": hi, "mean": round(mean, 12), "std": round(std, 12)},
                    )
                )
        return out


DEFAULT_CHECKS: tuple[type[Check], ...] = (
    IdColumnCheck,  # first: other checks use the IDs it finds
    TargetCopyCheck,
    NameTokenCheck,
    SingleFeatureCheck,
    DuplicateRowsCheck,
    GlobalNormalisationCheck,
)


def scan(
    table: pd.DataFrame,
    target: str,
    checks: tuple[type[Check], ...] = DEFAULT_CHECKS,
    context: Context | None = None,
) -> list[Finding]:
    """Run every check. A check that crashes raises; failures are never hidden."""
    if target not in table.columns:
        raise ValueError(f"Target column '{target}' not found")
    context = context or Context()
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check().run(table, target, context))
    # One row per column: keep the most severe finding first, the rest stay as evidence.
    order = {"block": 0, "warn": 1, "info": 2}
    return sorted(findings, key=lambda f: (order[f.severity], str(f.column)))


def flagged_columns(findings: list[Finding]) -> set[str]:
    return {f.column for f in findings if f.flagged and f.column is not None}
