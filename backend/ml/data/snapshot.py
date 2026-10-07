"""Data versions for live databases: what exactly a run trained on (#97).

A database changes while MLPilot works on it, so a run needs a record of the data it saw. Two modes:

* ``snapshot`` copies the tables a task needs (only the wanted columns, only rows with an event
  time up to ``as_of``) into the project's DuckDB file through guarded read-only ``SELECT``s. The
  version id is a hash of what was copied, so the same data is the same version and one changed
  row is another.
* ``live`` copies nothing. It records when the run started (``as_of``) and, per table, the row
  count and the latest event time, and says plainly that the same run cannot be reproduced
  exactly, because the source keeps changing.

``as_of`` defaults to the time of the call, in UTC. Event times are compared with it as UTC; a
time column without a time zone is taken to hold UTC.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, Field
from sqlglot import exp

from ml.data import sql_guard
from ml.data.engine import (
    INTERNAL_SCHEMA,
    EngineError,
    RowLimitExceeded,
    TableRef,
    quote_ident,
)
from ml.data.schema_graph import SchemaGraph, Table, type_kind
from ml.data.sources.base import SourceWithChecks
from ml.data.workspace import Workspace

SNAPSHOT_SCHEMA = "snapshots"
LIVE_NOTE = (
    "Live mode: the queries of this run read the source database as it is while they run. "
    "The same run cannot be reproduced exactly if the data changes."
)
SEPARATOR = "chr(31)"
NULL_MARK = "chr(0)"


class SnapshotError(EngineError):
    """A snapshot could not be taken as asked (never a silent partial copy)."""


@dataclass(frozen=True)
class SnapshotLimits:
    row_limit_per_table: int = 2_000_000
    max_bytes_per_table: int = 512 * 1024 * 1024
    timeout_s: float = 600.0


class TableRecord(BaseModel):
    """What was read from one table."""

    table: str = Field(description="The table's key in the schema graph")
    columns: list[str]
    time_column: str | None = Field(None, description="Rows after as_of were left out")
    rows: int
    max_event_time: str | None = Field(None, description="Latest event time among the rows, UTC")
    null_time_rows: int | None = Field(
        None,
        description="Rows without an event time, which cannot be placed in time and are left out",
    )
    checksum: str | None = Field(None, description="Order-independent hash of the copied rows")


class DataDescription(BaseModel):
    """Everything a manifest or report says about the data of a run on a database."""

    mode: Literal["snapshot", "live"]
    as_of: datetime
    connection: dict[str, str] = Field(default_factory=dict, description="Id and name; no secrets")
    tables: dict[str, TableRecord]
    reproducible: bool = Field(description="False in live mode")
    note: str | None = None

    @property
    def n_rows(self) -> int:
        return sum(t.rows for t in self.tables.values())

    @property
    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables.values())

    @property
    def max_event_times(self) -> dict[str, str | None]:
        return {k: t.max_event_time for k, t in self.tables.items()}


@dataclass(frozen=True)
class DataVersionInfo:
    id: str
    description: DataDescription
    created: bool  # False when the same data was already a version in this project

    @property
    def short_hash(self) -> str:
        return self.id[:12]


Progress = Callable[[float, str], None]


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _resolve_as_of(as_of: datetime | None) -> datetime:
    """The cutoff as an aware UTC time; a time without a zone is taken to be UTC."""
    if as_of is None:
        return now_utc()
    return as_of.astimezone(UTC) if as_of.tzinfo else as_of.replace(tzinfo=UTC)


def _naive_utc(as_of: datetime) -> datetime:
    return as_of.astimezone(UTC).replace(tzinfo=None) if as_of.tzinfo else as_of


# -- planning ------------------------------------------------------------------------------


@dataclass(frozen=True)
class TablePlan:
    table: Table
    columns: list[str]
    time_column: str | None
    time_kind: str | None  # timestamp, date or string
    time_tz: bool


def plan_tables(
    graph: SchemaGraph,
    dialect: str,
    tables: list[str] | None = None,
    columns: dict[str, list[str]] | None = None,
) -> list[TablePlan]:
    """The tables and columns to read, checked against the schema graph."""
    by_key = {t.key: t for t in graph.tables}
    wanted = tables if tables is not None else [t.key for t in graph.tables]
    if not wanted:
        raise SnapshotError("The database has no tables to read")
    plans = []
    for key in wanted:
        table = by_key.get(key)
        if table is None:
            raise SnapshotError(f"No table {key!r} in the schema (known: {sorted(by_key)})")
        available = [c.name for c in table.columns]
        chosen = (columns or {}).get(key) or available
        unknown = [c for c in chosen if c not in available]
        if unknown:
            raise SnapshotError(f"Table {key!r} has no column(s) {unknown}")
        time_col = table.time_column
        kind: str | None = None
        tz = False
        if time_col is not None:
            declared = next(c.type for c in table.columns if c.name == time_col)
            kind = type_kind(declared)
            tz = "with time zone" in declared.lower() or "timestamptz" in declared.lower()
            if kind not in ("timestamp", "date", "string") or (
                kind == "string" and dialect != "sqlite"
            ):
                raise SnapshotError(
                    f"The event-time column {key}.{time_col} has type {declared!r}. Snapshots need a "
                    "timestamp or date column (text only on SQLite). Choose another time column "
                    "or mark the table static."
                )
            if time_col not in chosen:
                chosen = [*chosen, time_col]
        plans.append(TablePlan(table, chosen, time_col, kind, tz))
    return plans


def _as_of_literal(dialect: str, plan: TablePlan, as_of: datetime) -> exp.Expression:
    text = _naive_utc(as_of).strftime("%Y-%m-%d %H:%M:%S")
    if dialect == "sqlite":
        return exp.Literal.string(text)
    if plan.time_tz:
        return exp.Cast(this=exp.Literal.string(f"{text}+00"), to=exp.DataType.build("timestamptz"))
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build("timestamp"))


def _table_expr(table: Table) -> exp.Table:
    return exp.table_(table.name, db=table.db_schema or None, quoted=True)


def select_sql(dialect: str, plan: TablePlan, as_of: datetime) -> str:
    """``SELECT <columns> FROM <table> WHERE <time> <= as_of`` (no filter for a static table)."""
    query = exp.select(*[exp.column(c, quoted=True) for c in plan.columns]).from_(
        _table_expr(plan.table)
    )
    if plan.time_column:
        query = query.where(
            exp.LTE(
                this=exp.column(plan.time_column, quoted=True),
                expression=_as_of_literal(dialect, plan, as_of),
            )
        )
    return query.sql(dialect=dialect)


def _scalar_sql(dialect: str, plan: TablePlan, as_of: datetime, what: str) -> str:
    """A guarded aggregate over the rows up to as_of: 'count', 'max_time' or 'null_time'."""
    table = _table_expr(plan.table)
    time_col = exp.column(plan.time_column, quoted=True) if plan.time_column else None
    if what == "null_time":
        assert time_col is not None
        query = (
            exp.select(exp.Count(this=exp.Star()))
            .from_(table)
            .where(exp.Is(this=time_col, expression=exp.Null()))
        )
    else:
        selects: list[exp.Expression] = [exp.Count(this=exp.Star())]
        if time_col is not None:
            selects.append(exp.Max(this=time_col))
        query = exp.select(*selects).from_(table)
        if time_col is not None:
            query = query.where(
                exp.LTE(this=time_col, expression=_as_of_literal(dialect, plan, as_of))
            )
    return query.sql(dialect=dialect)


# -- reading the source --------------------------------------------------------------------


def _read(source: SourceWithChecks, sql: str, limits: SnapshotLimits, limit: int) -> pa.Table:
    q = sql_guard.guard(sql, source.dialect, allowed_tables=sql_guard.allowed_tables(source))
    return sql_guard.execute(
        source, q, limit=limit, timeout_s=limits.timeout_s, max_bytes=limits.max_bytes_per_table
    )


def _event_time_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _naive_utc(value).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)[:19].replace("T", " ")


def _null_time_rows(
    source: SourceWithChecks, plan: TablePlan, as_of: datetime, limits: SnapshotLimits
) -> int | None:
    if plan.time_column is None:
        return None
    table = _read(source, _scalar_sql(source.dialect, plan, as_of, "null_time"), limits, 1)
    return int(table.column(0)[0].as_py())


def _event_time_sql(column: str, arrow_type: pa.DataType) -> str:
    """DuckDB expression: the column as a naive UTC timestamp."""
    col = quote_ident(column)
    if pa.types.is_timestamp(arrow_type):
        return f"CAST(timezone('UTC', {col}) AS TIMESTAMP)" if arrow_type.tz else col
    if pa.types.is_date(arrow_type):
        return f"CAST({col} AS TIMESTAMP)"
    return f"try_cast({col} AS TIMESTAMP)"


def _checksum_sql(ref: TableRef, columns: list[str]) -> str:
    # md5 of each row's text, summed: independent of row order, counts duplicates, and stable
    # across DuckDB versions (unlike its hash()).
    cells = ", ".join(f"coalesce(CAST({quote_ident(c)} AS VARCHAR), {NULL_MARK})" for c in columns)
    row = f"concat_ws({SEPARATOR}, {cells})"
    return f"SELECT coalesce(sum(md5_number_lower({row})::HUGEINT), 0) FROM {ref.sql()}"


def version_id_of(tables: list[TableRecord], scope: str = "") -> str:
    """SHA-256 of the copied data's identity: table, columns, row count, latest event time, checksum.

    ``scope`` (the project id) keeps two projects that read the same data from sharing one id, since
    version ids are primary keys in the metadata database."""
    payload = [
        {
            "table": t.table,
            "columns": t.columns,
            "rows": t.rows,
            "max_event_time": t.max_event_time,
            "checksum": t.checksum,
        }
        for t in sorted(tables, key=lambda t: t.table)
    ]
    body = json.dumps({"scope": scope, "tables": payload}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def snapshot_ref(version_id: str, table_key: str) -> TableRef:
    return TableRef(
        name=f"{version_id[:12]}__{table_key.replace('.', '__')}", schema=SNAPSHOT_SCHEMA
    )


def create_snapshot(
    source: SourceWithChecks,
    graph: SchemaGraph,
    workspace: Workspace,
    *,
    tables: list[str] | None = None,
    columns: dict[str, list[str]] | None = None,
    as_of: datetime | None = None,
    limits: SnapshotLimits | None = None,
    connection: dict[str, str] | None = None,
    progress: Progress | None = None,
    scope: str = "",
) -> DataVersionInfo:
    """Copy the wanted tables into the project's DuckDB file and return their data version.

    A table with more rows than ``limits.row_limit_per_table`` (or more bytes than the limit)
    fails the whole snapshot with the table's name; nothing partial is registered.
    """
    limits = limits or SnapshotLimits()
    as_of = _resolve_as_of(as_of)
    plans = plan_tables(graph, source.dialect, tables, columns)
    report = progress or (lambda _fraction, _message: None)
    duck = workspace.source
    token = uuid.uuid4().hex[:10]
    temp: dict[str, TableRef] = {}
    records: list[TableRecord] = []
    try:
        for i, plan in enumerate(plans):
            key = plan.table.key
            report(i / (len(plans) + 1), f"Reading {key}")
            try:
                data = _read(
                    source,
                    select_sql(source.dialect, plan, as_of),
                    limits,
                    limits.row_limit_per_table,
                )
            except RowLimitExceeded:
                raise SnapshotError(
                    f"Table {key!r} has more than {limits.row_limit_per_table:,} rows up to "
                    f"{as_of:%Y-%m-%d %H:%M:%S} UTC. Pick fewer or narrower tables, or sample "
                    "entities (planned), or raise the row limit."
                ) from None
            except sql_guard.ResultTooLarge:
                raise SnapshotError(
                    f"Table {key!r} is larger than {limits.max_bytes_per_table // (1024 * 1024)} MiB "
                    "in the columns asked for. Leave out wide columns or pick fewer tables."
                ) from None
            null_rows = _null_time_rows(source, plan, as_of, limits)
            ref = TableRef(name=f"tmp_{token}__{i}", schema=SNAPSHOT_SCHEMA)
            duck.register_arrow(data, ref.name, schema=ref.schema)
            temp[key] = ref
            records.append(_record_of(duck, plan, ref, data.schema, as_of, null_rows))
        report(len(plans) / (len(plans) + 1), "Fingerprinting the copy")
        version_id = version_id_of(records, scope)
        description = DataDescription(
            mode="snapshot",
            as_of=as_of,
            connection=connection or {},
            tables={r.table: r for r in records},
            reproducible=True,
        )
        if workspace.has_snapshot(version_id):
            return DataVersionInfo(version_id, description, created=False)
        final: dict[str, TableRef] = {}
        for key, ref in temp.items():
            final[key] = snapshot_ref(version_id, key)
            duck.execute(f"ALTER TABLE {ref.sql()} RENAME TO {quote_ident(final[key].name)}")
        workspace.register_snapshot(
            version_id, {key: (ref, description.tables[key].rows) for key, ref in final.items()}
        )
        temp.clear()
        report(1.0, "Snapshot ready")
        return DataVersionInfo(version_id, description, created=True)
    finally:
        for ref in temp.values():  # a failed or duplicate snapshot leaves nothing behind
            duck.execute(f"DROP TABLE IF EXISTS {ref.sql()}")


def _record_of(
    duck: Any,
    plan: TablePlan,
    ref: TableRef,
    arrow_schema: pa.Schema,
    as_of: datetime,
    null_rows: int | None,
) -> TableRecord:
    rows = int(duck.execute(f"SELECT count(*) FROM {ref.sql()}")[0][0])
    max_time: str | None = None
    if plan.time_column:
        expr = _event_time_sql(plan.time_column, arrow_schema.field(plan.time_column).type)
        latest, late = duck.execute(
            f"SELECT max({expr}), count(*) FILTER (WHERE {expr} > ?) FROM {ref.sql()}",
            [_naive_utc(as_of)],
        )[0]
        if late:  # whatever the source's text format was, the copy must hold nothing after as_of
            raise SnapshotError(
                f"{late} rows of {plan.table.key!r} have an event time after {as_of:%Y-%m-%d %H:%M:%S} UTC "
                f"in column {plan.time_column!r}; the snapshot was not kept."
            )
        max_time = _event_time_text(latest)
    checksum = str(duck.execute(_checksum_sql(ref, plan.columns))[0][0])
    return TableRecord(
        table=plan.table.key,
        columns=plan.columns,
        time_column=plan.time_column,
        rows=rows,
        max_event_time=max_time,
        null_time_rows=null_rows,
        checksum=checksum,
    )


def live_version(
    source: SourceWithChecks,
    graph: SchemaGraph,
    *,
    tables: list[str] | None = None,
    as_of: datetime | None = None,
    limits: SnapshotLimits | None = None,
    connection: dict[str, str] | None = None,
    scope: str = "",
) -> DataVersionInfo:
    """Record the state of the source at the start of a live-mode run. Reads two aggregates per table."""
    limits = limits or SnapshotLimits()
    as_of = _resolve_as_of(as_of)
    records = []
    for plan in plan_tables(graph, source.dialect, tables):
        result = _read(source, _scalar_sql(source.dialect, plan, as_of, "count"), limits, 1)
        latest = result.column(1)[0].as_py() if plan.time_column else None
        records.append(
            TableRecord(
                table=plan.table.key,
                columns=plan.columns,
                time_column=plan.time_column,
                rows=int(result.column(0)[0].as_py()),
                max_event_time=_event_time_text(latest),
                null_time_rows=_null_time_rows(source, plan, as_of, limits),
            )
        )
    description = DataDescription(
        mode="live",
        as_of=as_of,
        connection=connection or {},
        tables={r.table: r for r in records},
        reproducible=False,
        note=LIVE_NOTE,
    )
    # Each live run is its own version: it is identified by when it started.
    payload = json.dumps(
        {"scope": scope, "description": description.model_dump(mode="json")}, sort_keys=True
    )
    return DataVersionInfo(hashlib.sha256(payload.encode()).hexdigest(), description, created=True)


__all__ = [
    "INTERNAL_SCHEMA",
    "LIVE_NOTE",
    "SNAPSHOT_SCHEMA",
    "DataDescription",
    "DataVersionInfo",
    "SnapshotError",
    "SnapshotLimits",
    "TableRecord",
    "create_snapshot",
    "live_version",
    "plan_tables",
    "select_sql",
    "snapshot_ref",
    "version_id_of",
]
