"""One split contract for every experiment: which rows train, validate and test.

Tuning, early stopping, ensembles and keep/reject decisions only ever see the
train and validation rows. The test rows are scored once, at the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

if TYPE_CHECKING:
    from ml.tasks.spec import TaskSpec

# Below this many rows a single validation split is too noisy; use inner CV instead.
SMALL_DATA_ROWS = 2000


class SplitError(ValueError):
    """The split cannot be made as asked (never a silent empty or leaky split)."""


RANDOM_SPLIT_REFUSED = (
    "A relational task cannot use a random split: rows of the same entity at nearby cutoffs "
    "would land in both training and test, and a training label could cover the test period. "
    "Use the temporal split (train on earlier cutoffs, test on later ones)."
)


class SplitPlan(BaseModel):
    strategy: Literal["holdout", "cv", "temporal"] = "holdout"
    seed: int = 42
    test_fraction: float = Field(0.2, gt=0, lt=1)
    # holdout: share of all rows used for validation
    val_fraction: float = Field(0.2, gt=0, lt=1)
    # cv: number of inner folds over the non-test rows
    inner_folds: int | None = Field(None, ge=2)
    # temporal (filled in by the relational task work, #52)
    val_time: str | None = None
    test_time: str | None = None

    @classmethod
    def default_for(cls, n_rows: int, config: dict[str, Any] | None = None) -> SplitPlan:
        """Holdout for normal-sized data, inner 5-fold CV under SMALL_DATA_ROWS rows."""
        config = dict(config or {})
        # Older experiments used random_state / test_size / val_size.
        if "random_state" in config:
            config.setdefault("seed", config.pop("random_state"))
        if "test_size" in config:
            config.setdefault("test_fraction", config.pop("test_size"))
        if "val_size" in config:
            config.setdefault("val_fraction", config.pop("val_size"))
        known = {k: v for k, v in config.items() if k in cls.model_fields}
        if "strategy" not in known and n_rows < SMALL_DATA_ROWS:
            known.update(strategy="cv", inner_folds=known.get("inner_folds") or 5)
        if known.get("strategy") == "cv" and not known.get("inner_folds"):
            known["inner_folds"] = 5
        return cls(**known)


@dataclass
class SplitIndices:
    """Positional row indices. For cv, `train` holds all non-test rows and `folds` the inner folds."""

    plan: SplitPlan
    stratified: bool
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.plan.strategy,
            "seed": self.plan.seed,
            "stratified": self.stratified,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "inner_folds": len(self.folds) or None,
        }


def make_splits(
    y: pd.Series | np.ndarray, plan: SplitPlan, *, relational: bool = False
) -> SplitIndices:
    """Turn a target column and a plan into disjoint train/val/test row indices.

    Random splits (holdout, cv) are refused for relational tasks (``relational=True``): rows of
    one entity at nearby times would land on both sides, and a training label could cover the
    test period.
    """
    if plan.strategy == "temporal":
        raise SplitError(
            "A temporal split needs each row's cutoff and label window end: "
            "use make_temporal_splits"
        )
    if relational:
        raise SplitError(RANDOM_SPLIT_REFUSED)
    y = pd.Series(np.asarray(y))
    rows = np.arange(len(y))
    stratified = bool(y.value_counts().min() >= max(5, plan.inner_folds or 0))
    strat = y if stratified else None
    rest, test = train_test_split(
        rows, test_size=plan.test_fraction, random_state=plan.seed, stratify=strat
    )
    rest, test = np.sort(rest), np.sort(test)

    if plan.strategy == "cv":
        splitter = (
            StratifiedKFold(n_splits=plan.inner_folds, shuffle=True, random_state=plan.seed)
            if stratified
            else KFold(n_splits=plan.inner_folds, shuffle=True, random_state=plan.seed)
        )
        folds = [(rest[a], rest[b]) for a, b in splitter.split(rest, y.iloc[rest])]
        return SplitIndices(
            plan, stratified, train=rest, val=np.array([], dtype=int), test=test, folds=folds
        )

    val_share = plan.val_fraction / (1 - plan.test_fraction)
    train, val = train_test_split(
        rest,
        test_size=val_share,
        random_state=plan.seed,
        stratify=y.iloc[rest] if stratified else None,
    )
    return SplitIndices(plan, stratified, train=np.sort(train), val=np.sort(val), test=test)


# -- temporal splits for relational tasks (#52) -----------------------------------------------

Part = Literal["train", "val", "test", "purged_train", "purged_val"]


class TemporalSplitPlan(BaseModel):
    """Which cutoffs train, validate and test, by date. There is no seed: nothing is random."""

    strategy: Literal["temporal"] = "temporal"
    val_from: datetime
    test_from: datetime
    folds: int = Field(3, ge=2, description="Expanding-window folds over the training cutoffs")

    @classmethod
    def from_spec(cls, spec: TaskSpec, folds: int = 3) -> TemporalSplitPlan:
        return cls(
            val_from=_midnight(spec.split.val_from),
            test_from=_midnight(spec.split.test_from),
            folds=folds,
        )


def split_plan_for_task(spec: TaskSpec, requested: str | None = None) -> TemporalSplitPlan:
    """The plan a relational task runs with. Any other strategy is refused with a clear message."""
    if requested not in (None, "temporal"):
        raise SplitError(RANDOM_SPLIT_REFUSED)
    return TemporalSplitPlan.from_spec(spec)


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _naive_utc(value: Any) -> pd.Timestamp:
    """A time as naive UTC. A missing or unreadable time is an error, never a silent part."""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as e:
        raise SplitError(f"{value!r} is not a time") from e
    if ts is pd.NaT:
        raise SplitError("a cutoff or label window end is missing (NaT); fix or drop that row")
    return ts.tz_convert(UTC).tz_localize(None) if ts.tzinfo else ts


def _naive_utc_column(values: pd.Series | np.ndarray, name: str) -> pd.Series:
    """A column of times as naive UTC; naive values are read as UTC, offsets are converted."""
    try:
        parsed = pd.to_datetime(pd.Series(np.asarray(values)), utc=True)
    except (ValueError, TypeError) as e:
        raise SplitError(f"{name} holds a value that is not a time: {e}") from e
    if parsed.isna().any():
        raise SplitError(
            f"{name} has {int(parsed.isna().sum())} missing value(s); fix or drop them"
        )
    return parsed.dt.tz_localize(None)


def _plan_times(plan: TemporalSplitPlan) -> tuple[pd.Timestamp, pd.Timestamp]:
    val_from, test_from = _naive_utc(plan.val_from), _naive_utc(plan.test_from)
    if not val_from < test_from:
        raise SplitError("val_from must be before test_from")
    return val_from, test_from


def partition_cutoffs(
    pairs: list[tuple[Any, Any]], plan: TemporalSplitPlan
) -> dict[tuple[pd.Timestamp, pd.Timestamp], Part]:
    """The part of every (cutoff, label window end).

    * train: the cutoff is before ``val_from`` and its label window ends by ``val_from``, so no
      training label sees an event of the validation period;
    * val: the cutoff is in ``[val_from, test_from)`` and its window ends by ``test_from``;
    * test: the cutoff is at or after ``test_from``;
    * purged_train / purged_val: rows whose window crosses the next boundary. They are left out
      of fitting and scoring altogether, and reported.
    """
    val_from, test_from = _plan_times(plan)
    out: dict[tuple[pd.Timestamp, pd.Timestamp], Part] = {}
    for cutoff, window_end in pairs:
        c, w = _naive_utc(cutoff), _naive_utc(window_end)
        if w < c:
            raise SplitError(f"a label window ends ({w}) before its cutoff ({c})")
        part: Part
        if c >= test_from:
            part = "test"
        elif c >= val_from:
            part = "val" if w <= test_from else "purged_val"
        else:
            part = "train" if w <= val_from else "purged_train"
        out[(c, w)] = part
    return out


def expanding_folds(
    train_pairs: list[tuple[Any, Any]], folds: int
) -> list[tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]]]:
    """Expanding-window folds over the training cutoffs, with the same gap rule.

    The distinct cutoffs are cut into ``folds + 1`` consecutive blocks. Fold ``i`` validates on
    block ``i`` and trains on every row of an earlier block whose label window ends by the first
    cutoff of block ``i``.
    """
    pairs = sorted({(_naive_utc(c), _naive_utc(w)) for c, w in train_pairs})
    cutoffs = sorted({c for c, _ in pairs})
    if len(cutoffs) < folds + 1:
        raise SplitError(
            f"{folds} folds need at least {folds + 1} training cutoffs, there are {len(cutoffs)}: "
            "use fewer folds, more cutoffs or an earlier val_from"
        )
    blocks = [list(b) for b in np.array_split(np.array(cutoffs, dtype="datetime64[ns]"), folds + 1)]
    out = []
    for i in range(1, folds + 1):
        start = pd.Timestamp(blocks[i][0])
        in_block = {pd.Timestamp(c) for c in blocks[i]}
        val = [p for p in pairs if p[0] in in_block]
        train = [p for p in pairs if p[0] < start and p[1] <= start]
        if not train:
            raise SplitError(
                f"fold {i} has no training rows: label windows of the earliest cutoffs end after "
                f"{start}. Use fewer folds, a longer training range or a shorter horizon"
            )
        out.append((train, val))
    return out


class RowSplit(Protocol):
    """What the run loop needs of a split, temporal or random: positional row indices.

    ``train`` and ``val`` fit and early-stop the champion, ``test`` is scored once, and
    ``folds`` (each with ``train`` and ``val`` rows) are the paired folds of the acceptance rule,
    drawn from ``fold_rows``.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def fold_rows(self) -> np.ndarray: ...

    @property
    def folds(self) -> Sequence[Any]: ...


@dataclass
class TemporalFold:
    train: np.ndarray
    val: np.ndarray
    val_start: pd.Timestamp
    val_end: pd.Timestamp


@dataclass
class TemporalSplit:
    """Positional row indices of a temporal split."""

    plan: TemporalSplitPlan
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    purged_train: np.ndarray
    purged_val: np.ndarray
    folds: list[TemporalFold]
    parts: dict[tuple[pd.Timestamp, pd.Timestamp], Part]
    max_train_window_end: pd.Timestamp
    max_val_window_end: pd.Timestamp | None
    entity_overlap: dict[str, int] | None = None

    @property
    def fold_rows(self) -> np.ndarray:
        """The rows the acceptance folds are drawn from: the training rows."""
        return self.train

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": "temporal",
            "val_from": self.plan.val_from.isoformat(),
            "test_from": self.plan.test_from.isoformat(),
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_purged_train": len(self.purged_train),
            "n_purged_val": len(self.purged_val),
            "max_train_window_end": self.max_train_window_end.isoformat(),
            "max_val_window_end": (
                self.max_val_window_end.isoformat() if self.max_val_window_end is not None else None
            ),
            "folds": [
                {
                    "n_train": len(f.train),
                    "n_val": len(f.val),
                    "val_from": f.val_start.isoformat(),
                    "val_to": f.val_end.isoformat(),
                }
                for f in self.folds
            ],
            "entity_overlap": self.entity_overlap,
        }


def timeline(
    parts: dict[tuple[pd.Timestamp, pd.Timestamp], Part], rows: dict[Any, int] | None = None
) -> list[dict[str, Any]]:
    """One entry per (cutoff, window end), in time order: for the run view."""
    out = []
    for (c, w), part in sorted(parts.items()):
        out.append(
            {
                "cutoff": c.isoformat(),
                "window_end": w.isoformat(),
                "part": part,
                "rows": (rows or {}).get((c, w)),
            }
        )
    return out


def make_temporal_splits(
    cutoff_time: pd.Series | np.ndarray,
    window_end: pd.Series | np.ndarray,
    plan: TemporalSplitPlan,
    entity_id: pd.Series | np.ndarray | None = None,
) -> TemporalSplit:
    """Split label rows by cutoff date, never at random.

    ``cutoff_time`` and ``window_end`` hold one value per row (the ``cutoff_time`` and
    ``label_window_end`` columns of the label table). Raises ``SplitError`` if train, validation
    or test would be empty. With ``entity_id`` the split also reports how many entities appear in
    more than one part (expected for relational tasks, but the report shows it).
    """
    cut = _naive_utc_column(cutoff_time, "cutoff_time")
    end = _naive_utc_column(window_end, "window_end")
    if len(cut) != len(end):
        raise SplitError("cutoff_time and window_end need the same length")
    pairs = list(zip(cut, end, strict=True))
    parts = partition_cutoffs(sorted(set(pairs)), plan)
    row_part = pd.Series([parts[p] for p in pairs])

    def idx(part: Part) -> np.ndarray:
        return np.flatnonzero((row_part == part).to_numpy())

    train, val, test = idx("train"), idx("val"), idx("test")
    for name, rows in (("training", train), ("validation", val), ("test", test)):
        if len(rows) == 0:
            raise SplitError(
                f"No {name} rows: check val_from, test_from, the horizon and the cutoff range"
            )
    folds = []
    for fold_train, fold_val in expanding_folds(
        [p for p in parts if parts[p] == "train"], plan.folds
    ):
        in_train = set(fold_train)
        in_val = set(fold_val)
        folds.append(
            TemporalFold(
                train=np.flatnonzero(np.array([p in in_train for p in pairs])),
                val=np.flatnonzero(np.array([p in in_val for p in pairs])),
                val_start=min(c for c, _ in fold_val),
                val_end=max(c for c, _ in fold_val),
            )
        )
    overlap = None
    if entity_id is not None:
        ent = np.asarray(entity_id)
        e_train, e_val, e_test = set(ent[train]), set(ent[val]), set(ent[test])
        overlap = {
            "train_and_val": len(e_train & e_val),
            "train_and_test": len(e_train & e_test),
            "val_and_test": len(e_val & e_test),
            "entities_in_train": len(e_train),
            "entities_in_test": len(e_test),
        }
    return TemporalSplit(
        plan=plan,
        train=train,
        val=val,
        test=test,
        purged_train=idx("purged_train"),
        purged_val=idx("purged_val"),
        folds=folds,
        parts=parts,
        max_train_window_end=end.iloc[train].max(),
        max_val_window_end=end.iloc[val].max(),
        entity_overlap=overlap,
    )
