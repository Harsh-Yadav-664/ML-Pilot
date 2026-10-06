"""Column statistics for tables of a connected database (issue #96).

The CSV path profiles a pandas frame. A live table can have 50 million rows, so here the
statistics are computed by the database itself, as aggregate SQL through the SQL guard:

* one aggregate query per table (per ``columns_per_query`` columns on very wide tables) that
  returns exactly one row: non-null count, distinct count, min / max / mean / stddev /
  p1 / p50 / p99 for numbers, min / max for times, the true share for booleans;
* for categorical text columns one more query per column returning at most ``top_k``
  (value, count) pairs, so the top values are aggregates too, never rows;
* tables above ``sample_above_rows`` rows are profiled on a sample (``TABLESAMPLE SYSTEM`` on
  Postgres, ``USING SAMPLE ... BERNOULLI`` on DuckDB, ``rowid % k`` on SQLite) and say so:
  ``sampled``, ``sample_fraction`` and ``sample_method`` are part of the result.

Every query goes through ``sql_guard.guard`` (restricted to the tables of the database) and
``sql_guard.execute`` with a row limit and a statement timeout, and the whole table has a
time budget. A failure raises; it is never turned into empty statistics.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ml.data import sql_guard
from ml.data.engine import EngineError, QueryTimeout, TableRef, quote_ident
from ml.data.schema_graph import (
    Kind,
    id_stem,
    table_key,
    table_row_count,
    type_kind,
)
from ml.data.sources.base import SourceWithChecks

logger = logging.getLogger(__name__)

SemanticType = Literal["id", "categorical", "numeric", "time", "text", "boolean", "other"]
SAMPLE_SEED = 42
QUANTILES = {"p1": 0.01, "p50": 0.5, "p99": 0.99}
DOUBLE = "DOUBLE PRECISION"


class StatsConfig(BaseModel):
    sample_above_rows: int = Field(1_000_000, description="Tables with more rows are sampled")
    sample_rows: int = Field(1_000_000, description="Rows a sample aims for")
    timeout_s: float = Field(60.0, description="Statement timeout of every query")
    budget_s: float = Field(300.0, description="Time budget of the whole table")
    top_k: int = Field(20, ge=1, le=100)
    categorical_max_distinct: int = Field(50, description="Text columns up to this are categorical")
    columns_per_query: int = Field(40, ge=1, description="Postgres allows 1664 select items")


class TopValue(BaseModel):
    value: str
    count: int


class ColumnStats(BaseModel):
    name: str
    type: str
    semantic_type: SemanticType
    non_null: int
    null_fraction: float
    distinct: int = Field(description="Distinct values among the profiled rows")
    min: float | None = None
    max: float | None = None
    mean: float | None = Field(None, description="Booleans: the share of true")
    stddev: float | None = Field(None, description="Sample standard deviation")
    p1: float | None = None
    p50: float | None = None
    p99: float | None = None
    time_min: str | None = None
    time_max: str | None = None
    top_values: list[TopValue] = Field(
        default_factory=list,
        description="Categorical columns only. These are cell values: they reach an LLM only "
        "where the project allows category labels (#48)",
    )


class TableStats(BaseModel):
    table: str
    row_count: int
    row_count_estimated: bool = False
    profiled_rows: int
    sampled: bool
    sample_fraction: float | None = None
    sample_method: str | None = Field(
        None, description="system, bernoulli, rowid_modulo or head; None when not sampled"
    )
    seconds: float
    columns: list[ColumnStats]


# -- semantic types ----------------------------------------------------------------------


def semantic_type(
    name: str,
    kind: Kind,
    *,
    is_key: bool,
    non_null: int,
    distinct: int,
    minimum: float | None,
    maximum: float | None,
    categorical_max_distinct: int = 50,
) -> SemanticType:
    """The semantic type of a column from its declared type and its statistics."""
    if kind in ("timestamp", "date"):
        return "time"
    if kind == "boolean":
        return "boolean"
    keyish = is_key or name.lower() == "id" or id_stem(name) is not None
    if kind == "integer" and keyish:
        return "id"
    if kind == "integer":
        if minimum == 0 and maximum == 1 and distinct <= 2:
            return "boolean"
        return "numeric"
    if kind == "float":
        return "numeric"
    if kind == "text":
        if keyish and non_null and distinct / non_null >= 0.5:
            return "id"
        if distinct and distinct <= categorical_max_distinct:
            return "categorical"
        return "text"
    return "other"


# -- the queries -------------------------------------------------------------------------


class _Run:
    """Runs guarded queries for one table against one time budget."""

    def __init__(self, source: SourceWithChecks, config: StatsConfig) -> None:
        self.source = source
        self.config = config
        self.allowed = sql_guard.allowed_tables(source)
        self.started = time.monotonic()
        self.queries = 0

    def rows(self, sql: str, *, limit: int) -> list[dict[str, Any]]:
        elapsed = time.monotonic() - self.started
        if elapsed > self.config.budget_s:
            raise QueryTimeout(f"Profiling ran over its {self.config.budget_s:g} s budget")
        timeout = min(self.config.timeout_s, self.config.budget_s - elapsed)
        query = sql_guard.guard(sql, self.source.dialect, allowed_tables=self.allowed)
        table = sql_guard.execute(self.source, query, limit=limit, timeout_s=timeout)
        self.queries += 1
        return table.to_pylist()

    def one_row(self, sql: str) -> dict[str, Any]:
        (row,) = self.rows(sql, limit=1)  # the guard raises if the database returns more
        return row


def _from_clause(
    ref: TableRef, dialect: str, row_count: int, config: StatsConfig, method: str | None
) -> tuple[str, str | None, float | None]:
    """(a FROM item that yields the profiled rows, sample method, sample fraction)."""
    table = f"{quote_ident(ref.schema)}.{quote_ident(ref.name)}"
    if row_count <= config.sample_above_rows:
        return f"(SELECT * FROM {table}) AS s", None, None
    fraction = min(1.0, config.sample_rows / row_count)
    if dialect == "postgres":
        sample = f"TABLESAMPLE SYSTEM ({fraction * 100:.6f}) REPEATABLE ({SAMPLE_SEED})"
        return f"(SELECT * FROM {table} {sample}) AS s", "system", fraction
    if dialect == "duckdb":
        sample = f"USING SAMPLE {fraction * 100:.6f} PERCENT (bernoulli, {SAMPLE_SEED})"
        return f"(SELECT * FROM {table} {sample}) AS s", "bernoulli", fraction
    modulus = math.ceil(row_count / config.sample_rows)
    if method == "head":
        return f"(SELECT * FROM {table} LIMIT {config.sample_rows}) AS s", "head", fraction
    return (
        f"(SELECT * FROM {table} WHERE rowid % {modulus} = 0) AS s",
        "rowid_modulo",
        1 / modulus,
    )


def _numeric(col: str) -> str:
    return f"CAST({quote_ident(col)} AS {DOUBLE})"


def _aggregates(columns: list[tuple[int, str, Kind]], dialect: str) -> list[str]:
    items: list[str] = []
    for i, name, kind in columns:
        c = quote_ident(name)
        items += [f"COUNT({c}) AS nn_{i}", f"COUNT(DISTINCT {c}) AS d_{i}"]
        if kind in ("integer", "float"):
            x = _numeric(name)
            items += [f"MIN({x}) AS mn_{i}", f"MAX({x}) AS mx_{i}", f"AVG({x}) AS av_{i}"]
            if dialect != "sqlite":
                items.append(f"STDDEV_SAMP({x}) AS sd_{i}")
                for label, q in QUANTILES.items():
                    if dialect == "postgres":
                        items.append(
                            f"PERCENTILE_CONT({q}) WITHIN GROUP (ORDER BY {x}) AS {label}_{i}"
                        )
                    else:
                        items.append(f"QUANTILE_CONT({x}, {q}) AS {label}_{i}")
        elif kind in ("timestamp", "date"):
            items += [f"MIN({c}) AS tmn_{i}", f"MAX({c}) AS tmx_{i}"]
        elif kind == "boolean":
            items.append(f"AVG(CAST(CASE WHEN {c} THEN 1 ELSE 0 END AS {DOUBLE})) AS av_{i}")
    return items


def _sqlite_extras(
    run: _Run, from_item: str, columns: list[tuple[int, str, Kind]], first: dict[str, Any]
) -> dict[str, Any]:
    """SQLite has no STDDEV or quantile aggregates: the standard deviation is a second pass
    around the mean, and a quantile is read at its (interpolated) position in the sort."""
    items: list[str] = []
    plan: dict[str, tuple[int, int, float]] = {}
    for i, name, kind in columns:
        if kind not in ("integer", "float"):
            continue
        n, mean, x, c = first[f"nn_{i}"], first[f"av_{i}"], _numeric(name), quote_ident(name)
        if n >= 2 and mean is not None:
            m = repr(float(mean))
            items.append(f"(SELECT SUM(({x} - {m}) * ({x} - {m})) FROM {from_item}) AS ss_{i}")
        for label, q in QUANTILES.items():
            if not n:
                continue
            position = (n - 1) * q
            low, high = math.floor(position), math.ceil(position)
            plan[f"{label}_{i}"] = (low, high, position - low)
            for offset in {low, high}:
                items.append(
                    f"(SELECT {x} FROM {from_item} WHERE {c} IS NOT NULL "
                    f"ORDER BY {c} LIMIT 1 OFFSET {offset}) AS q_{i}_{label}_{offset}"
                )
    if not items:
        return {}
    row = run.one_row(f"SELECT {', '.join(items)}")
    out: dict[str, Any] = {}
    for i, name, kind in columns:
        if kind not in ("integer", "float"):
            continue
        n = first[f"nn_{i}"]
        if f"ss_{i}" in row and row[f"ss_{i}"] is not None:
            out[f"sd_{i}"] = math.sqrt(row[f"ss_{i}"] / (n - 1))
        for label in QUANTILES:
            if f"{label}_{i}" in plan:
                low, high, frac = plan[f"{label}_{i}"]
                a, b = row[f"q_{i}_{label}_{low}"], row[f"q_{i}_{label}_{high}"]
                out[f"{label}_{i}"] = a + (b - a) * frac
    return out


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def profile_table(
    source: SourceWithChecks, ref: TableRef, config: StatsConfig | None = None
) -> TableStats:
    """Statistics of one table, computed in the database. Raises on any failure."""
    config = config or StatsConfig()
    started = time.monotonic()
    run = _Run(source, config)
    schema = source.table_schema(ref)
    key_columns = set(schema.primary_key) | {c for fk in schema.foreign_keys for c in fk.columns}
    row_count, estimated = table_row_count(source, ref)
    columns = [(i, c.name, type_kind(c.type)) for i, c in enumerate(schema.columns)]
    types = {c.name: c.type for c in schema.columns}

    method: str | None = None
    first: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    total = 0
    sample_method: str | None = None
    fraction: float | None = None
    for start in range(0, len(columns), config.columns_per_query):
        chunk = columns[start : start + config.columns_per_query]
        while True:
            from_item, sample_method, fraction = _from_clause(
                ref, source.dialect, row_count, config, method
            )
            items = ["COUNT(*) AS total", *_aggregates(chunk, source.dialect)]
            try:
                row = run.one_row(f"SELECT {', '.join(items)} FROM {from_item}")
                break
            except EngineError as e:
                # A SQLite table WITHOUT ROWID cannot be sampled by rowid: say so and use the
                # first rows instead (recorded in sample_method), rather than failing.
                if (
                    source.dialect == "sqlite"
                    and sample_method == "rowid_modulo"
                    and "rowid" in str(e)
                ):
                    method = "head"
                    continue
                raise
        total = int(row["total"])
        first.update(row)
        if source.dialect == "sqlite":
            extras.update(_sqlite_extras(run, from_item, chunk, row))

    stats: list[ColumnStats] = []
    for i, name, kind in columns:
        non_null, distinct = int(first[f"nn_{i}"]), int(first[f"d_{i}"])
        minimum, maximum = _float(first.get(f"mn_{i}")), _float(first.get(f"mx_{i}"))
        sem = semantic_type(
            name,
            kind,
            is_key=name in key_columns,
            non_null=non_null,
            distinct=distinct,
            minimum=minimum,
            maximum=maximum,
            categorical_max_distinct=config.categorical_max_distinct,
        )
        values = {**first, **extras}
        column = ColumnStats(
            name=name,
            type=types[name],
            semantic_type=sem,
            non_null=non_null,
            null_fraction=round(1 - non_null / total, 12) if total else 0.0,
            distinct=distinct,
            min=minimum,
            max=maximum,
            mean=_float(values.get(f"av_{i}")),
            stddev=_float(values.get(f"sd_{i}")),
            p1=_float(values.get(f"p1_{i}")),
            p50=_float(values.get(f"p50_{i}")),
            p99=_float(values.get(f"p99_{i}")),
            time_min=None if values.get(f"tmn_{i}") is None else str(values[f"tmn_{i}"]),
            time_max=None if values.get(f"tmx_{i}") is None else str(values[f"tmx_{i}"]),
        )
        if sem == "categorical":
            column.top_values = _top_values(run, from_item, name, config.top_k)
        stats.append(column)

    seconds = time.monotonic() - started
    logger.info(
        "profiled table rows=%d sampled=%s queries=%d seconds=%.2f",
        row_count,
        sample_method is not None,
        run.queries,
        seconds,
    )
    return TableStats(
        table=table_key(ref),
        row_count=row_count,
        row_count_estimated=estimated,
        profiled_rows=total,
        sampled=sample_method is not None,
        sample_fraction=fraction,
        sample_method=sample_method,
        seconds=round(seconds, 3),
        columns=stats,
    )


def _top_values(run: _Run, from_item: str, column: str, k: int) -> list[TopValue]:
    c = quote_ident(column)
    rows = run.rows(
        f"SELECT {c} AS value, COUNT(*) AS n FROM {from_item} WHERE {c} IS NOT NULL "
        f"GROUP BY {c} ORDER BY n DESC, {c} LIMIT {k}",
        limit=k,
    )
    return [TopValue(value=str(r["value"]), count=int(r["n"])) for r in rows]
