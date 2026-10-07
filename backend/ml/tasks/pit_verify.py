"""Runtime check for a feature query (#51): recompute it on data cut off at each cutoff.

``pit_guard.check`` reads the SQL. This module checks the *result*, as defence in depth: for a
sample of label rows, the feature is computed on the full data and again on a copy in which every
row at or after the cutoff has been deleted. A feature that stays inside its cutoff gives the same
value both times. A different value is a leak the SQL analysis missed, and it is reported with the
entity, the cutoff and both values.

It also checks the contract: the query returns ``entity_id, cutoff_time, value`` with at most one
row per label row, and no rows that are not label rows.

The query must already have passed ``sql_guard.guard`` (a single read-only SELECT). It runs on
in-memory copies of the data in a DuckDB connection with no access to files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from ml.data.engine import quote_ident
from ml.data.schema_graph import SchemaGraph, type_kind
from ml.tasks.pit_guard import CUTOFF, ENTITY, LABELS, OUTPUT_COLUMNS, VALUE

TIME_ZONE = "UTC"


class ContractError(ValueError):
    """The feature query does not return what the contract asks for."""


@dataclass(frozen=True)
class Mismatch:
    entity_id: Any
    cutoff: datetime
    full: Any
    truncated: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "cutoff": self.cutoff.isoformat(),
            "full": self.full,
            "truncated": self.truncated,
        }


@dataclass
class TruncationResult:
    ok: bool
    checked_rows: int
    cutoffs: list[datetime]
    mismatches: list[Mismatch] = field(default_factory=list)


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(
        ":memory:",
        config={
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        },
    )
    con.execute(f"SET TimeZone = '{TIME_ZONE}'")
    return con


def _labels_frame(labels: pd.DataFrame) -> pd.DataFrame:
    missing = {ENTITY, CUTOFF} - set(labels.columns)
    if missing:
        raise ContractError(f"the labels need the columns {ENTITY} and {CUTOFF}")
    frame = labels[[ENTITY, CUTOFF]].copy()
    frame[CUTOFF] = pd.to_datetime(frame[CUTOFF])
    return frame.drop_duplicates().reset_index(drop=True)


def _load(
    con: duckdb.DuckDBPyConnection,
    tables: dict[str, pa.Table],
    graph: SchemaGraph,
    cutoff: datetime | None,
) -> None:
    """Make every table visible under its name; with ``cutoff``, only its rows before it."""
    by_name = {t.name.lower(): t for t in graph.tables}
    for name, data in tables.items():
        raw = f"{name}__raw"
        con.register(raw, data)
        table = by_name.get(name.lower())
        where = ""
        if cutoff is not None and table is not None and table.time_column and not table.is_static:
            stamp = cutoff.strftime("%Y-%m-%d %H:%M:%S.%f")
            declared = next(c.type for c in table.columns if c.name == table.time_column)
            if type_kind(declared) == "date":  # no time of day: the cutoff's day is deleted too
                where = (
                    f" WHERE CAST({quote_ident(table.time_column)} AS DATE) "
                    f"< CAST(TIMESTAMP '{stamp}' AS DATE)"
                )
            else:
                where = (
                    f" WHERE TRY_CAST({quote_ident(table.time_column)} AS TIMESTAMP) "
                    f"< TIMESTAMP '{stamp}'"
                )
        con.execute(f"CREATE VIEW {quote_ident(name)} AS SELECT * FROM {quote_ident(raw)}{where}")


def run_feature(
    sql: str,
    tables: dict[str, pa.Table],
    labels: pd.DataFrame,
    graph: SchemaGraph,
    *,
    cutoff: datetime | None = None,
) -> pd.DataFrame:
    """Run a feature query over ``tables``, with ``labels`` as ``__labels``.

    With ``cutoff``, rows at or after it are deleted from every table that has an event time.
    Raises ``ContractError`` if the result is not ``entity_id, cutoff_time, value`` with unique
    keys that all belong to the labels.
    """
    frame = _labels_frame(labels)
    con = _connect()
    try:
        _load(con, tables, graph, cutoff)
        con.register(LABELS, frame)
        out = con.execute(sql).fetch_df()
    finally:
        con.close()
    if tuple(c.lower() for c in out.columns) != OUTPUT_COLUMNS:
        raise ContractError(
            f"the query returned {list(out.columns)}, not {list(OUTPUT_COLUMNS)} in that order"
        )
    out.columns = list(OUTPUT_COLUMNS)
    out[CUTOFF] = pd.to_datetime(out[CUTOFF])
    if out.duplicated([ENTITY, CUTOFF]).any():
        raise ContractError("the query returns more than one row for a label row")
    keys = set(zip(frame[ENTITY], frame[CUTOFF], strict=True))
    extra = [k for k in zip(out[ENTITY], out[CUTOFF], strict=True) if k not in keys]
    if extra:
        raise ContractError(f"the query returns rows that are not label rows, such as {extra[0]}")
    return out


def _same(a: Any, b: Any) -> bool:
    a_null = a is None or (isinstance(a, float) and math.isnan(a)) or a is pd.NA or a is pd.NaT
    b_null = b is None or (isinstance(b, float) and math.isnan(b)) or b is pd.NA or b is pd.NaT
    if a_null or b_null:
        return a_null and b_null
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return bool(a == b)


def _pick_cutoffs(frame: pd.DataFrame, limit: int) -> list[Any]:
    """Half of the limit from the cutoffs with most label rows, the rest spread over the others."""
    counts = frame.groupby(CUTOFF).size().sort_values(ascending=False, kind="stable")
    distinct = sorted(counts.index)
    if len(distinct) <= limit:
        return distinct
    busiest = list(counts.index[: max(1, limit // 2)])
    rest = [c for c in distinct if c not in busiest]
    need = limit - len(busiest)
    step = (len(rest) - 1) / (need - 1) if need > 1 else 0
    spread = [rest[round(i * step)] for i in range(need)] if need else []
    return sorted({*busiest, *spread})


def truncation_check(
    sql: str,
    tables: dict[str, pa.Table],
    graph: SchemaGraph,
    labels: pd.DataFrame,
    *,
    max_cutoffs: int = 5,
    rows_per_cutoff: int = 200,
    seed: int = 0,
) -> TruncationResult:
    """Compare the feature on full data with the feature on data cut at each cutoff.

    Takes up to ``max_cutoffs`` cutoffs (the busiest ones and others spread over the range) and
    up to ``rows_per_cutoff`` label rows of each. ``ok`` is False if any value differs.
    """
    frame = _labels_frame(labels)
    cutoffs = _pick_cutoffs(frame, max_cutoffs)
    sample = pd.concat(
        [
            frame[frame[CUTOFF] == c].sample(
                n=min(rows_per_cutoff, int((frame[CUTOFF] == c).sum())), random_state=seed
            )
            for c in cutoffs
        ]
    )
    full = run_feature(sql, tables, sample, graph)
    full_values = {(r[ENTITY], r[CUTOFF]): r[VALUE] for _, r in full.iterrows()}
    mismatches: list[Mismatch] = []
    checked = 0
    for c in cutoffs:
        part = sample[sample[CUTOFF] == c]
        cut = run_feature(sql, tables, part, graph, cutoff=pd.Timestamp(c).to_pydatetime())
        cut_values = {(r[ENTITY], r[CUTOFF]): r[VALUE] for _, r in cut.iterrows()}
        for key in zip(part[ENTITY], part[CUTOFF], strict=True):
            checked += 1
            a, b = full_values.get(key), cut_values.get(key)
            if not _same(a, b):
                mismatches.append(Mismatch(key[0], pd.Timestamp(key[1]).to_pydatetime(), a, b))
    return TruncationResult(
        not mismatches, checked, [pd.Timestamp(c).to_pydatetime() for c in cutoffs], mismatches
    )


def tables_from_source(source: Any, names: list[str]) -> dict[str, pa.Table]:
    """Whole tables as Arrow, read through a source's own ``query`` (the SQL guard)."""
    return {
        n: source.query(f"SELECT * FROM {quote_ident(n)}", limit=10**8, timeout_s=300)
        for n in names
    }
