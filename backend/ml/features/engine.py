"""Run feature queries for a whole label table at once (#57): one query per feature, cached.

The tables of a data version (a snapshot, #97) and the label rows are loaded once into an
in-memory DuckDB. A feature is one query over every label row, not one query per row. Its values
are cached as Parquet under a key made of the normalised query, the data version, the label table
version and the entity sample, so the same feature is computed once across repeated runs. A query
that runs past its time limit, or that breaks the contract, is recorded as ``failed`` with the
reason, and the caller carries on with the next feature.

Large tasks are sampled: when (entities x cutoffs) is above ``max_rows``, a fixed-seed sample of
entities is kept, stratified by whether the entity ever had a positive label, and the fraction is
recorded. Every cutoff of a kept entity is kept, so a row never loses its history.

Budgets (``Budget``, ``BudgetTracker``) cap the cost of the language model, the wall time and the
number of proposals of one run; ``propose_within_budget`` stops at the first one that is reached,
after the step in progress, and says which.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlglot
from sqlglot.errors import SqlglotError

from ml.data.schema_graph import SchemaGraph
from ml.tasks.pit_guard import CUTOFF, ENTITY, VALUE
from ml.tasks.pit_verify import ContractError, FeatureSession, FeatureTimeout

DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_TIMEOUT_S = 120.0
LABEL = "label"
MAX_REASON = 300


def sql_hash(sql: str, dialect: str = "duckdb") -> str:
    """A hash of the query with its text normalised (case, spacing, quoting)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        text = tree.sql(dialect=dialect, normalize=True)
    except SqlglotError:
        text = " ".join(sql.lower().split())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Sampling:
    """How the label table was cut down: ``fraction`` of the entities, ``seed`` for the draw."""

    sampled: bool
    fraction: float
    entities_kept: int
    entities_total: int
    rows_kept: int
    rows_total: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_entities(
    labels: pd.DataFrame, max_rows: int = DEFAULT_MAX_ROWS, seed: int = 42
) -> tuple[pd.DataFrame, Sampling]:
    """Keep all rows of a fixed-seed sample of entities when there are more than ``max_rows`` rows.

    The sample is stratified by whether the entity has any positive row (a binary label) so the
    rare class is kept in proportion. Rows keep their order.
    """
    total_rows = len(labels)
    entities = labels[ENTITY].drop_duplicates()
    if total_rows <= max_rows:
        return labels, Sampling(
            False, 1.0, len(entities), len(entities), total_rows, total_rows, seed
        )
    fraction = max_rows / total_rows
    if LABEL in labels.columns:
        positive = labels.groupby(ENTITY)[LABEL].max().gt(0)
    else:
        positive = pd.Series(False, index=entities.to_numpy())
    rng = np.random.default_rng(seed)
    kept: list[Any] = []
    for flag in (True, False):
        group = sorted(positive.index[positive == flag], key=str)
        take = round(len(group) * fraction)
        if take:
            kept.extend(rng.choice(np.array(group, dtype=object), size=take, replace=False))
    out = labels[labels[ENTITY].isin(set(kept))].reset_index(drop=True)
    return out, Sampling(True, fraction, len(kept), len(entities), len(out), total_rows, seed)


FeatureStatus = Literal["computed", "cached", "failed"]


@dataclass
class Computed:
    sql_hash: str
    status: FeatureStatus
    seconds: float
    values: pd.Series | None = field(repr=False, default=None)  # one per label row, in order
    reason: str | None = None  # "timeout: ...", "contract: ...", "error: ..."

    @property
    def ok(self) -> bool:
        return self.values is not None


class FeatureEngine:
    """Computes feature queries over one label table and caches the values."""

    def __init__(
        self,
        graph: SchemaGraph,
        tables: dict[str, pa.Table],
        labels: pd.DataFrame,
        *,
        data_version_id: str,
        label_version: str,
        cache_dir: Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_rows: int = DEFAULT_MAX_ROWS,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.labels, self.sampling = sample_entities(labels, max_rows, seed)
        self.labels = self.labels.reset_index(drop=True)
        self.index = pd.MultiIndex.from_frame(
            self.labels[[ENTITY, CUTOFF]].assign(**{CUTOFF: pd.to_datetime(self.labels[CUTOFF])})
        )
        # the label rows themselves are part of the key: new cutoffs under the same names must miss
        row_hash = int(pd.util.hash_pandas_object(self.index.to_frame(index=False)).sum())
        self._context = "|".join(
            [
                data_version_id,
                label_version,
                f"{self.sampling.fraction:.6f}",
                str(seed),
                f"{len(self.index)}:{row_hash}",
            ]
        )
        self._session = FeatureSession(tables, self.labels, graph)
        self.counts = {"computed": 0, "cached": 0, "failed": 0}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def cache_key(self, sql: str) -> str:
        return hashlib.sha256(f"{sql_hash(sql)}|{self._context}".encode()).hexdigest()[:32]

    def _path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.parquet"

    def _align(self, frame: pd.DataFrame) -> pd.Series:
        frame = frame.assign(**{CUTOFF: pd.to_datetime(frame[CUTOFF])})
        values = frame.set_index([ENTITY, CUTOFF])[VALUE].reindex(self.index)
        column = values if pd.api.types.is_numeric_dtype(values) else values.astype("float64")
        return column.reset_index(drop=True)

    def compute(self, sql: str, *, timeout_s: float | None = None) -> Computed:
        key = self.cache_key(sql)
        digest = sql_hash(sql)
        path = self._path(key)
        started = time.monotonic()
        if path is not None and path.exists():
            values = self._align(pq.read_table(path).to_pandas())
            self.counts["cached"] += 1
            return Computed(digest, "cached", time.monotonic() - started, values)
        limit = self.timeout_s if timeout_s is None else timeout_s
        try:
            frame = self._session.run(sql, timeout_s=limit)
        except FeatureTimeout as e:
            return self._failed(digest, started, f"timeout: {e}")
        except ContractError as e:
            return self._failed(digest, started, f"contract: {e}")
        except duckdb.Error as e:
            return self._failed(digest, started, f"error: {_clip(str(e))}")
        try:
            values = self._align(frame)
        except (ValueError, TypeError) as e:  # for example a text value where a number is needed
            return self._failed(
                digest, started, f"error: the values are not numbers: {_clip(str(e))}"
            )
        if path is not None:
            tmp = path.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp)
            tmp.replace(path)
        self.counts["computed"] += 1
        return Computed(digest, "computed", time.monotonic() - started, values)

    def _failed(self, digest: str, started: float, reason: str) -> Computed:
        self.counts["failed"] += 1
        return Computed(digest, "failed", time.monotonic() - started, None, reason)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_REASON else text[: MAX_REASON - 1] + "…"


# -- budgets -------------------------------------------------------------------------------------

BudgetName = Literal["cost", "time", "proposals"]


@dataclass(frozen=True)
class Budget:
    """Limits of one run; ``None`` means no limit. A feature's own time limit is the engine's."""

    max_cost_usd: float | None = None
    max_seconds: float | None = None
    max_proposals: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_cost_usd": self.max_cost_usd,
            "max_seconds": self.max_seconds,
            "max_proposals": self.max_proposals,
        }


class BudgetTracker:
    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self.cost_usd = 0.0
        self.proposals = 0

    @property
    def seconds(self) -> float:
        return self._clock() - self._started

    def charge(self, cost_usd: float) -> None:
        self.cost_usd += cost_usd

    def exceeded(self) -> BudgetName | None:
        """The first limit that is reached (spent >= limit), or None."""
        b = self.budget
        if b.max_cost_usd is not None and self.cost_usd >= b.max_cost_usd:
            return "cost"
        if b.max_seconds is not None and self.seconds >= b.max_seconds:
            return "time"
        if b.max_proposals is not None and self.proposals >= b.max_proposals:
            return "proposals"
        return None

    def used(self) -> dict[str, Any]:
        return {
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 3),
            "proposals": self.proposals,
        }


class Events(Protocol):
    async def emit(self, type_: str, **payload: Any) -> int: ...


@dataclass
class BudgetedRun:
    """What ``propose_within_budget`` returns. ``stopped`` names the budget that ended it."""

    records: list[Any]
    stopped: BudgetName | None
    used: dict[str, Any]

    @property
    def status(self) -> str:
        return f"stopped: budget ({self.stopped})" if self.stopped else "completed"


class _Proposer(Protocol):
    async def propose(self, remaining: int = 1) -> Any: ...


async def propose_within_budget(
    proposer: _Proposer,
    count: int,
    tracker: BudgetTracker,
    events: Events | None = None,
) -> BudgetedRun:
    """Ask for up to ``count`` features, one at a time, until a budget is reached.

    The budget is checked before each proposal and charged after it (a proposal can use two model
    calls, the second being the repair round), so the step in progress always finishes: the run
    can go over a limit by at most one proposal. A proposal whose feature query timed out or
    failed is just a rejected proposal; the loop goes on with the next one. If no real model
    answers the loop ends at once.
    """
    records: list[Any] = []
    stopped: BudgetName | None = None
    for i in range(count):
        stopped = tracker.exceeded()
        if stopped:
            break
        record = await proposer.propose(count - i)
        tracker.proposals += 1
        cost = sum(float(call.get("cost_usd") or 0.0) for call in record.llm)
        tracker.charge(cost)
        records.append(record)
        if events is not None:
            await events.emit(
                "feature_proposal",
                index=i + 1,
                status=record.status,
                stage=record.stage,
                cost_usd=round(cost, 6),
                budget_used=tracker.used(),
            )
        if record.status == "no_llm":
            break
    else:
        stopped = None
    if stopped and events is not None:
        await events.emit(
            "budget_stop", budget=stopped, limits=tracker.budget.as_dict(), used=tracker.used()
        )
    return BudgetedRun(records, stopped, tracker.used())
