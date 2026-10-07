"""Is this prediction task worth training? Checks on the label table before any model runs (#98).

A model trained on 40 positives, or on a base rate that jumps from 2% to 30% between cutoffs, gives
numbers that mean nothing, and the user would not know. Every check returns ``ok``, ``warn`` or
``block`` with the numbers it used. A ``block`` stops a run from starting unless the caller says
``override`` (the override is recorded with the report).

The checks read only what the label builder already produced: the counts per cutoff (#50), the
split (#52) and, for coverage, how many entities have rows in each related table before the first
cutoff. Thresholds are a ``Thresholds`` value, so a project can change them; the values used are
stored in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ml.tasks.labels import CutoffCount, TableCoverage
from ml.tasks.spec import TaskSpec
from ml.validation.splits import SplitError, TemporalSplitPlan, partition_cutoffs

Status = Literal["ok", "warn", "block"]
_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "block": 2}


class Thresholds(BaseModel):
    """The limits the checks use. A value at the limit passes; below it (or above, for ratios) fails."""

    min_train_positives: int = Field(50, ge=0, description="Below this: block")
    min_eval_positives: int = Field(20, ge=0, description="Validation or test, below this: block")
    warn_positives: int = Field(200, ge=0, description="Any split below this: warn")
    max_base_rate_ratio: float = Field(3.0, gt=1, description="Highest / lowest cutoff base rate")
    max_eligible_drop: float = Field(0.5, gt=0, lt=1, description="Fall between adjacent cutoffs")
    imbalance_rate: float = Field(0.01, gt=0, lt=0.5, description="Rarer class below this: warn")
    min_coverage: float = Field(0.05, ge=0, le=1, description="Best related table below this: warn")


@dataclass(frozen=True)
class Check:
    code: str
    status: Status
    message: str
    numbers: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "message": self.message,
            "numbers": self.numbers,
        }


@dataclass(frozen=True)
class FeasibilityReport:
    status: Status
    checks: list[Check]
    thresholds: Thresholds
    metric: str
    suggested_metric: str | None

    @property
    def blocked(self) -> bool:
        return self.status == "block"

    @property
    def reasons(self) -> list[str]:
        """What stops the run, or, if nothing does, what the user should look at."""
        worst = [c for c in self.checks if c.status == "block"] or [
            c for c in self.checks if c.status == "warn"
        ]
        return [c.message for c in worst]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blocked": self.blocked,
            "reasons": self.reasons,
            "checks": [c.as_dict() for c in self.checks],
            "thresholds": self.thresholds.model_dump(),
            "metric": self.metric,
            "suggested_metric": self.suggested_metric,
        }


def _naive(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _positives_per_split(spec: TaskSpec, counts: list[CutoffCount], t: Thresholds) -> Check:
    code = "positives_per_split"
    if spec.target.type != "binary":
        return Check(code, "ok", "Not a binary task: positives are not counted", {})
    pairs = [(_naive(c.cutoff), _naive(c.window_end)) for c in counts]
    try:
        plan = TemporalSplitPlan.from_spec(spec)
        parts = partition_cutoffs(pairs, plan)
    except SplitError as e:
        return Check(code, "block", f"The split cannot be made: {e}", {})
    totals = {"train": 0, "val": 0, "test": 0}
    for c, pair in zip(counts, pairs, strict=True):
        part = parts[pair]
        if part in totals:
            totals[part] += c.positives or 0
    numbers = {f"{k}_positives": v for k, v in totals.items()}
    short = [
        name
        for name, n, need in (
            ("training", totals["train"], t.min_train_positives),
            ("validation", totals["val"], t.min_eval_positives),
            ("test", totals["test"], t.min_eval_positives),
        )
        if n < need
    ]
    if short:
        return Check(
            code,
            "block",
            f"Too few positives to train or judge a model: {', '.join(short)} "
            f"({totals['train']} train, {totals['val']} validation, {totals['test']} test; "
            f"need at least {t.min_train_positives} train and {t.min_eval_positives} in each of "
            "validation and test)",
            numbers,
        )
    if min(totals.values()) < t.warn_positives:
        return Check(
            code,
            "warn",
            f"Few positives: {totals['train']} train, {totals['val']} validation, "
            f"{totals['test']} test (below {t.warn_positives} in at least one). "
            "Scores will be noisy",
            numbers,
        )
    return Check(
        code,
        "ok",
        f"{totals['train']} train, {totals['val']} validation, {totals['test']} test positives",
        numbers,
    )


def _base_rate_per_cutoff(spec: TaskSpec, counts: list[CutoffCount], t: Thresholds) -> Check:
    code = "base_rate_per_cutoff"
    if spec.target.type != "binary":
        return Check(code, "ok", "Not a binary task: no base rate", {})
    rated = [c for c in counts if c.eligible > 0 and c.base_rate is not None]
    if not rated:
        return Check(code, "warn", "No cutoff has eligible entities", {})
    empty = [c for c in rated if (c.positives or 0) == 0]
    rates = [float(c.base_rate) for c in rated if c.base_rate]
    low, high = (min(rates), max(rates)) if rates else (0.0, 0.0)
    numbers = {
        "min_base_rate": low,
        "max_base_rate": high,
        "cutoffs_without_positives": [c.cutoff.date().isoformat() for c in empty],
    }
    if empty:
        return Check(
            code,
            "warn",
            f"{len(empty)} cutoff(s) have no positives (first: {empty[0].cutoff:%Y-%m-%d}). "
            "Either the label definition or the data has a problem there",
            numbers,
        )
    if low > 0 and high / low > t.max_base_rate_ratio:
        return Check(
            code,
            "warn",
            f"The base rate moves from {_pct(low)} to {_pct(high)} between cutoffs "
            f"(a factor of {high / low:.1f}, limit {t.max_base_rate_ratio:g}). "
            "A model trained on one period may not hold in another",
            numbers,
        )
    return Check(code, "ok", f"Base rate between {_pct(low)} and {_pct(high)} per cutoff", numbers)


def _eligible_drops(counts: list[CutoffCount], t: Thresholds) -> Check:
    code = "eligible_per_cutoff"
    ordered = sorted(counts, key=lambda c: c.cutoff)
    drops = []
    for before, after in pairwise(ordered):
        if before.eligible > 0 and (before.eligible - after.eligible) / before.eligible > (
            t.max_eligible_drop
        ):
            drops.append((before, after))
    numbers = {
        "min_eligible": min((c.eligible for c in counts), default=0),
        "max_eligible": max((c.eligible for c in counts), default=0),
        "drops": [
            {
                "from": b.cutoff.date().isoformat(),
                "to": a.cutoff.date().isoformat(),
                "before": b.eligible,
                "after": a.eligible,
            }
            for b, a in drops
        ],
    }
    if drops:
        b, a = drops[0]
        return Check(
            code,
            "warn",
            f"Eligible entities fall from {b.eligible} to {a.eligible} between "
            f"{b.cutoff:%Y-%m-%d} and {a.cutoff:%Y-%m-%d} (more than "
            f"{_pct(t.max_eligible_drop)}): check for a gap in the data",
            numbers,
        )
    return Check(
        code,
        "ok",
        f"Eligible entities per cutoff: {numbers['min_eligible']} to {numbers['max_eligible']}",
        numbers,
    )


def _imbalance(
    spec: TaskSpec, counts: list[CutoffCount], t: Thresholds
) -> tuple[Check, str | None]:
    code = "class_imbalance"
    if spec.target.type != "binary":
        return Check(code, "ok", "Not a binary task: no class balance", {}), None
    eligible = sum(c.eligible for c in counts)
    positives = sum(c.positives or 0 for c in counts)
    if eligible == 0:
        return Check(code, "warn", "No eligible entities", {}), None
    rate = positives / eligible
    rarer = min(rate, 1 - rate)
    numbers = {"base_rate": rate, "positives": positives, "eligible": eligible}
    if rarer < t.imbalance_rate:
        metric = spec.metric_name
        advice = (
            "PR-AUC is the metric"
            if metric == "pr_auc"
            else f"PR-AUC suits this better than {metric}"
        )
        return (
            Check(
                code,
                "warn",
                f"The rarer class is {_pct(rarer)} of rows (base rate {_pct(rate)}); {advice}. "
                "Lift at k is not available yet (#93)",
                numbers,
            ),
            "pr_auc",
        )
    return Check(code, "ok", f"Base rate {_pct(rate)}", numbers), None


def _coverage(coverage: list[TableCoverage] | None, t: Thresholds) -> Check:
    code = "coverage"
    if coverage is None:
        return Check(code, "ok", "Coverage was not computed", {})
    if not coverage:
        return Check(
            code,
            "warn",
            "No table with an event time and a link to the entity: there is nothing to build "
            "features from",
            {"tables": {}},
        )
    shares = {c.table: c.share for c in coverage}
    best = max(coverage, key=lambda c: c.share)
    numbers = {"tables": shares, "entities_at_first_cutoff": best.eligible}
    if best.share < t.min_coverage:
        return Check(
            code,
            "warn",
            f"At the first cutoff, at most {_pct(best.share)} of entities have any earlier row in "
            f"a related table (best: {best.table}); features will be mostly empty",
            numbers,
        )
    return Check(
        code,
        "ok",
        f"At the first cutoff, {_pct(best.share)} of entities have earlier rows in "
        f"{best.table}, the best-covered related table",
        numbers,
    )


def _horizon(dropped: int, kept: int) -> Check:
    code = "horizon_vs_data"
    numbers = {"cutoffs_kept": kept, "cutoffs_dropped": dropped}
    if dropped:
        return Check(
            code,
            "warn",
            f"{dropped} cutoff(s) were dropped because their label window ends after the data "
            "does; labels are never filled in for them",
            numbers,
        )
    return Check(code, "ok", "Every cutoff has a complete label window", numbers)


def check_feasibility(
    spec: TaskSpec,
    counts: list[CutoffCount],
    *,
    dropped_cutoffs: int = 0,
    coverage: list[TableCoverage] | None = None,
    thresholds: Thresholds | None = None,
) -> FeasibilityReport:
    """Run every check on the label counts of a task. ``counts`` has one entry per kept cutoff."""
    t = thresholds or Thresholds()
    imbalance, suggested = _imbalance(spec, counts, t)
    checks = [
        _positives_per_split(spec, counts, t),
        _base_rate_per_cutoff(spec, counts, t),
        _eligible_drops(counts, t),
        imbalance,
        _coverage(coverage, t),
        _horizon(dropped_cutoffs, len(counts)),
    ]
    status: Status = max((c.status for c in checks), key=lambda s: _ORDER[s])
    return FeasibilityReport(status, checks, t, spec.metric_name, suggested)
