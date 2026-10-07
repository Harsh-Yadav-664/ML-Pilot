"""Labels at cutoff dates: one row per (entity, cutoff), the label read only from the future window (#50).

``compile_labels`` turns a validated task spec into one SELECT (built with sqlglot, rendered for
DuckDB or Postgres):

* ``cutoffs``: the schedule, each cutoff with its window end, as UNION ALL of literals. Cutoffs
  whose window ends after the data does are left out here and reported, never kept with an
  incomplete label.
* ``eligible``: every entity at every cutoff where it existed before the cutoff
  (``created_at < cutoff``), every eligibility condition holds, and every ``exists`` or
  ``not_exists`` test holds over that table's rows strictly before the cutoff. Those tests always
  get the time bound ``event_time < cutoff`` added by the compiler, so an eligibility rule cannot
  read the future even if the spec forgot to say so.
* ``agg``: the target table LEFT JOINed on the entity key and limited to the window
  ``(cutoff, window_end]`` (an event at exactly the cutoff is not in the window, one at exactly
  the window end is), aggregated per (entity, cutoff). The LEFT JOIN keeps entities with no
  events, so a count is 0 and a sum is 0 for them.

The result is ``entity_id, cutoff_time, label, label_window_end``. Time is compared as UTC; a
column without a time zone is taken to hold UTC.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlglot import exp

from ml.data import sql_guard
from ml.data.engine import INTERNAL_SCHEMA, TableRef, arrow_to_pandas, quote_ident
from ml.data.schema_graph import Column, Edge, SchemaGraph, Table, type_kind
from ml.data.snapshot import DataDescription
from ml.data.sources import SourceWithChecks
from ml.data.workspace import Workspace
from ml.tasks.conditions import parse_condition
from ml.tasks.spec import (
    ExistsRule,
    TaskSpec,
    add_duration,
    cutoff_dates,
    parse_duration,
    validate_against,
)

CUTOFF_ALIAS = "c"
ENTITY_ALIAS = "e"
TARGET_ALIAS = "t"
_COMPARE = re.compile(r"^(=|==|!=|<>|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$")
_WINDOW_OFFSET = re.compile(
    r"^:cutoff(?:\s*\+\s*interval\s*'(\d+)\s*(hour|day|week)s?')?$", re.IGNORECASE
)
_HOURS = {"hour": 1, "day": 24, "week": 168}


class LabelError(ValueError):
    """The spec cannot be turned into labels as asked (never a silent partial result)."""


@dataclass(frozen=True)
class CutoffWindow:
    cutoff: datetime
    window_end: datetime

    def as_dict(self) -> dict[str, str]:
        return {"cutoff": self.cutoff.isoformat(), "window_end": self.window_end.isoformat()}


@dataclass(frozen=True)
class CompiledLabels:
    select: exp.Select
    dialect: str
    sql: str  # pretty-printed, for the user
    kept: list[CutoffWindow]
    dropped: list[CutoffWindow]  # window ends after the data does


TableResolver = Callable[[str], exp.Table]


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def window_end_of(spec: TaskSpec, cutoff: date) -> datetime:
    """When the label window of ``cutoff`` closes: the horizon, or an earlier ``window.end``."""
    window = spec.target.window
    if window is not None and window.end is not None:
        m = _WINDOW_OFFSET.match(window.end.strip())
        if m is None or m.group(1) is None:
            raise LabelError(f"cannot read target.window.end {window.end!r}")
        return _midnight(cutoff) + timedelta(hours=int(m.group(1)) * _HOURS[m.group(2).lower()])
    return _midnight(add_duration(cutoff, parse_duration(spec.horizon)))


def window_start_of(spec: TaskSpec, cutoff: date) -> datetime:
    window = spec.target.window
    if window is not None:
        m = _WINDOW_OFFSET.match(window.start.strip())
        if m is not None and m.group(1) is not None:
            return _midnight(cutoff) + timedelta(hours=int(m.group(1)) * _HOURS[m.group(2).lower()])
    return _midnight(cutoff)


def label_windows(spec: TaskSpec, as_of: datetime) -> tuple[list[CutoffWindow], list[CutoffWindow]]:
    """(kept, dropped): cutoffs whose label window is complete by ``as_of``, and those that are not."""
    as_of = as_of.astimezone(UTC) if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    kept: list[CutoffWindow] = []
    dropped: list[CutoffWindow] = []
    for d in cutoff_dates(spec):
        w = CutoffWindow(_midnight(d), window_end_of(spec, d))
        (kept if w.window_end <= as_of else dropped).append(w)
    return kept, dropped


# -- what the data must contain -------------------------------------------------------------


def _table(graph: SchemaGraph, name: str) -> Table:
    by_key = {t.key: t for t in graph.tables}
    if name in by_key:
        return by_key[name]
    same = [t for t in graph.tables if t.name == name]
    if len(same) == 1:
        return same[0]
    raise LabelError(f"table {name!r} is not in the schema")


def _link(graph: SchemaGraph, child: Table, entity: Table, via: str | None) -> Edge:
    edges = [e for e in graph.edges if e.from_table == child.key and e.to_table == entity.key]
    if via is not None:
        edges = [e for e in edges if e.from_columns == [via]]
    if len(edges) != 1 or len(edges[0].from_columns) != 1 or len(edges[0].to_columns) != 1:
        raise LabelError(f"{child.key!r} needs exactly one single-column key to {entity.key!r}")
    return edges[0]


def required_columns(spec: TaskSpec, graph: SchemaGraph) -> dict[str, list[str]]:
    """The columns of each table the labels read, to ask a snapshot for (#97)."""
    need: dict[str, set[str]] = {}
    entity = _table(graph, spec.entity.table)
    need.setdefault(entity.key, set()).add(spec.entity.key)
    if spec.entity.created_at:
        need[entity.key].add(spec.entity.created_at)

    def add_child(child: Table, via: str | None, where: str | None) -> None:
        edge = _link(graph, child, entity, via)
        cols = need.setdefault(child.key, set())
        cols.add(edge.from_columns[0])
        if child.time_column:
            cols.add(child.time_column)
        if where:
            cols.update(parse_condition(where, table=child.name).columns)

    for rule in spec.eligibility:
        if isinstance(rule, str):
            need[entity.key].update(parse_condition(rule, table=entity.name).columns)
        else:
            test = rule.exists or rule.not_exists
            assert test is not None
            add_child(_table(graph, test.table), test.via, test.where)
    ex = spec.target.expression
    if ex is not None:
        child = _table(graph, ex.table)
        add_child(child, ex.via, ex.where)
        if ex.column:
            need[child.key].add(ex.column)
    return {k: sorted(v) for k, v in need.items()}


# -- compiling ------------------------------------------------------------------------------


def _col(alias: str, name: str) -> exp.Column:
    return exp.column(name, table=alias, quoted=True)


def _stamp(dialect: str, value: datetime, *, tz: bool) -> exp.Expression:
    text = value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    if tz:
        return exp.Cast(this=exp.Literal.string(f"{text}+00"), to=exp.DataType.build("timestamptz"))
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build("timestamp"))


def _tz_aware(declared: str) -> bool:
    lowered = declared.lower()
    return "time zone" in lowered or "timestamptz" in lowered


def _is_temporal(col: Column) -> bool:
    return col.hint == "time" or type_kind(col.type) in ("timestamp", "date")


def _time_expr(dialect: str, alias: str, table: Table, name: str | None = None) -> exp.Expression:
    """A time column of ``table`` (default: its event time) as a naive UTC timestamp.

    Time-zone-aware times are converted to UTC, so they compare with the cutoffs the same way
    whatever the session's time zone is. Text times (SQLite) are parsed, and in DuckDB, which
    reads snapshots that keep whatever type the source delivered, every time column is cast.
    """
    name = name or table.time_column
    assert name is not None
    col = next(c for c in table.columns if c.name == name)
    column = _col(alias, name)
    if _tz_aware(col.type):
        return exp.AtTimeZone(this=column, zone=exp.Literal.string("UTC"))
    if type_kind(col.type) == "text" or (dialect == "duckdb" and _is_temporal(col)):
        return exp.TryCast(this=column, to=exp.DataType.build("timestamp"))
    return column


def _render_condition(dialect: str, text: str, table: Table, alias: str) -> exp.Expression:
    cond = parse_condition(text, table=table.name)
    by_name = {c.name: c for c in table.columns}

    def swap(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Placeholder):
            return _col(CUTOFF_ALIAS, "cutoff_time")
        if isinstance(node, exp.Column):
            col = by_name.get(node.name)
            if col is not None and _is_temporal(col):
                return _time_expr(dialect, alias, table, node.name)
            return _col(alias, node.name)
        return node

    return cond.tree.copy().transform(swap)


def _and(parts: list[exp.Expression]) -> exp.Expression | None:
    out: exp.Expression | None = None
    for part in parts:
        wrapped = exp.paren(part) if isinstance(part, exp.Or) else part
        out = wrapped if out is None else exp.And(this=out, expression=wrapped)
    return out


def _compare(compare: str) -> tuple[str, str]:
    m = _COMPARE.match(compare.strip())
    if m is None:
        raise LabelError(f"cannot read compare {compare!r}")
    op = "=" if m.group(1) == "==" else m.group(1)
    return op, m.group(2)


_OPS: dict[str, type[exp.Binary]] = {
    "=": exp.EQ,
    "!=": exp.NEQ,
    "<>": exp.NEQ,
    ">": exp.GT,
    ">=": exp.GTE,
    "<": exp.LT,
    "<=": exp.LTE,
}


def compile_labels(
    spec: TaskSpec,
    graph: SchemaGraph,
    dialect: str,
    resolve: TableResolver,
    as_of: datetime,
) -> CompiledLabels:
    """The label query for ``spec``. ``resolve(table key)`` gives the table to read it from."""
    if dialect not in ("duckdb", "postgres"):
        raise LabelError(
            f"labels are compiled for duckdb and postgres, not {dialect}; "
            "take a snapshot to run them on DuckDB"
        )
    problems = [
        i
        for i in validate_against(spec, graph, as_of, drop_incomplete_cutoffs=True)
        if i.severity == "error"
    ]
    if problems:
        raise LabelError(
            "The spec has errors: " + "; ".join(f"{p.path}: {p.message}" for p in problems[:5])
        )
    if spec.target.type != "binary" and spec.target.type != "regression":
        raise LabelError(f"{spec.target.type} labels are not supported yet")
    if spec.target.expression_sql is not None:
        raise LabelError(
            "labels from target.expression_sql are not compiled here; they need the "
            "point-in-time guard (#51). Use target.expression"
        )
    ex = spec.target.expression
    assert ex is not None
    kept, dropped = label_windows(spec, as_of)
    if not kept:
        raise LabelError(
            "Every cutoff's label window ends after the data does, so no labels are complete. "
            "Move the cutoffs back or use newer data"
        )
    entity = _table(graph, spec.entity.table)
    target = _table(graph, ex.table)

    # cutoffs ---------------------------------------------------------------------------------
    cutoff_rows = [
        exp.select(
            exp.alias_(_stamp(dialect, w.cutoff, tz=False), "cutoff_time"),
            exp.alias_(_stamp(dialect, w.window_end, tz=False), "window_end"),
        )
        for w in kept
    ]
    cutoffs: exp.Query = cutoff_rows[0]
    for row in cutoff_rows[1:]:
        cutoffs = exp.union(cutoffs, row, distinct=False)

    # eligible --------------------------------------------------------------------------------
    key = _col(ENTITY_ALIAS, spec.entity.key)
    cutoff_time = _col(CUTOFF_ALIAS, "cutoff_time")
    conditions: list[exp.Expression] = []
    if spec.entity.created_at:
        conditions.append(
            exp.LT(
                this=_time_expr(dialect, ENTITY_ALIAS, entity, spec.entity.created_at),
                expression=cutoff_time,
            )
        )
    for rule in spec.eligibility:
        if isinstance(rule, str):
            conditions.append(_render_condition(dialect, rule, entity, ENTITY_ALIAS))
            continue
        kind_exists = rule.exists is not None
        test = (rule.exists if kind_exists else rule.not_exists) or ExistsRule().exists
        assert test is not None
        child = _table(graph, test.table)
        edge = _link(graph, child, entity, test.via)
        inner: list[exp.Expression] = [
            exp.EQ(
                this=_col("x", edge.from_columns[0]),
                expression=_col(ENTITY_ALIAS, edge.to_columns[0]),
            )
        ]
        if child.time_column:  # always before the cutoff, whatever the spec says
            inner.append(exp.LT(this=_time_expr(dialect, "x", child), expression=cutoff_time))
        if test.where:
            inner.append(_render_condition(dialect, test.where, child, "x"))
        sub = (
            exp.select(exp.Literal.number(1))
            .from_(exp.alias_(resolve(child.key), "x"))
            .where(_and(inner))
        )
        exists_node: exp.Expression = exp.Exists(this=sub)
        conditions.append(exists_node if kind_exists else exp.Not(this=exists_node))
    eligible = (
        exp.select(
            exp.alias_(key, "entity_id"),
            exp.alias_(cutoff_time, "cutoff_time"),
            exp.alias_(_col(CUTOFF_ALIAS, "window_end"), "window_end"),
        )
        .from_(exp.alias_(resolve(entity.key), ENTITY_ALIAS))
        .join(exp.alias_(exp.to_table("cutoffs"), CUTOFF_ALIAS), join_type="cross")
    )
    where = _and(conditions)
    if where is not None:
        eligible = eligible.where(where)

    # agg: the window is (cutoff, window_end] --------------------------------------------------
    edge = _link(graph, target, entity, ex.via)
    fk = _col(TARGET_ALIAS, edge.from_columns[0])
    t_time = _time_expr(dialect, TARGET_ALIAS, target)
    start_time: exp.Expression = _col("el", "cutoff_time")
    offset = window_start_of(spec, cutoff_dates(spec)[0]) - _midnight(cutoff_dates(spec)[0])
    if offset > timedelta(0):
        hours = int(offset.total_seconds() // 3600)
        start_time = exp.Add(
            this=start_time,
            expression=exp.Interval(this=exp.Literal.string(str(hours)), unit=exp.var("HOUR")),
        )
    on: list[exp.Expression] = [
        exp.EQ(this=fk, expression=_col("el", "entity_id")),
        exp.GT(this=t_time, expression=start_time),
        exp.LTE(
            this=_time_expr(dialect, TARGET_ALIAS, target), expression=_col("el", "window_end")
        ),
    ]
    if ex.where:
        on.append(_render_condition(dialect, ex.where, target, TARGET_ALIAS))
    value_col = _col(TARGET_ALIAS, ex.column) if ex.column else fk
    if ex.agg == "count":
        value: exp.Expression = exp.Count(this=value_col)
    elif ex.agg == "count_distinct":
        value = exp.Count(this=exp.Distinct(expressions=[value_col]))
    elif ex.agg == "sum":
        value = exp.Coalesce(this=exp.Sum(this=value_col), expressions=[exp.Literal.number(0)])
    elif ex.agg == "avg":
        value = exp.Avg(this=value_col)
    elif ex.agg == "min":
        value = exp.Min(this=value_col)
    else:
        value = exp.Max(this=value_col)
    agg = (
        exp.select(
            exp.alias_(_col("el", "entity_id"), "entity_id"),
            exp.alias_(_col("el", "cutoff_time"), "cutoff_time"),
            exp.alias_(_col("el", "window_end"), "window_end"),
            exp.alias_(value, "value"),
        )
        .from_(exp.alias_(exp.to_table("eligible"), "el"))
        .join(exp.alias_(resolve(target.key), TARGET_ALIAS), on=_and(on), join_type="left")
        .group_by(_col("el", "entity_id"), _col("el", "cutoff_time"), _col("el", "window_end"))
    )

    # the label ---------------------------------------------------------------------------------
    v = _col("a", "value")
    label: exp.Expression
    if spec.target.type == "binary":
        assert ex.compare is not None
        op, number = _compare(ex.compare)
        compared = _OPS[op](this=v, expression=exp.Literal.number(number))
        label = exp.Case(
            ifs=[
                exp.If(this=exp.Is(this=v, expression=exp.Null()), true=exp.Null()),
                exp.If(this=compared, true=exp.Literal.number(1)),
            ],
            default=exp.Literal.number(0),
        )
    else:
        label = v
    final = (
        exp.select(
            exp.alias_(_col("a", "entity_id"), "entity_id"),
            exp.alias_(_col("a", "cutoff_time"), "cutoff_time"),
            exp.alias_(label, "label"),
            exp.alias_(_col("a", "window_end"), "label_window_end"),
        )
        .from_(exp.alias_(exp.to_table("agg"), "a"))
        .where(exp.Not(this=exp.Is(this=v, expression=exp.Null())))
        .with_("cutoffs", as_=cutoffs)
        .with_("eligible", as_=eligible)
        .with_("agg", as_=agg)
    )
    return CompiledLabels(
        select=final,
        dialect=dialect,
        sql=final.sql(dialect=dialect, pretty=True),
        kept=kept,
        dropped=dropped,
    )


def counts_select(labels: exp.Select, *, binary: bool) -> exp.Select:
    """Per cutoff: eligible entities, and for binary tasks positives and the base rate."""
    sub = exp.alias_(exp.Subquery(this=labels.copy()), "l")
    selects: list[exp.Expr] = [
        exp.alias_(_col("l", "cutoff_time"), "cutoff_time"),
        exp.alias_(exp.Count(this=exp.Star()), "eligible"),
    ]
    if binary:
        selects.append(exp.alias_(exp.Sum(this=_col("l", "label")), "positives"))
    else:
        selects.append(exp.alias_(exp.Avg(this=_col("l", "label")), "mean_label"))
    return (
        exp.select(*selects)
        .from_(sub)
        .group_by(_col("l", "cutoff_time"))
        .order_by(_col("l", "cutoff_time"))
    )


def describe(compiled: CompiledLabels) -> dict[str, Any]:
    return {
        "sql": compiled.sql,
        "cutoffs": [w.as_dict() for w in compiled.kept],
        "dropped_cutoffs": [w.as_dict() for w in compiled.dropped],
    }


def coverage_select(
    spec: TaskSpec,
    graph: SchemaGraph,
    compiled: CompiledLabels,
    resolve: TableResolver,
    held: dict[str, list[str]] | None = None,
) -> tuple[exp.Select, list[str]] | None:
    """One query: of the entities at the first cutoff, how many have rows in each related table.

    A related table is any table with an event time and a direct foreign key to the entity table.
    A row counts only if its event time is before the cutoff, so this reads no future data.
    ``held`` (a snapshot's tables and columns) limits it to what the snapshot contains.
    Returns the query and the table names in column order, or None if there is no related table.
    """
    entity = _table(graph, spec.entity.table)
    first = compiled.kept[0].cutoff
    cutoff_time = _col("l", "cutoff_time")
    names: list[str] = []
    selects: list[exp.Expr] = [exp.alias_(exp.Count(this=exp.Star()), "eligible")]
    for table in graph.tables:
        if table.key == entity.key or not table.time_column or table.is_static:
            continue
        edges = [e for e in graph.edges if e.from_table == table.key and e.to_table == entity.key]
        if not edges:
            continue
        edge = edges[0]
        if held is not None and not {edge.from_columns[0], table.time_column} <= set(
            held.get(table.key, [])
        ):
            continue
        inner = (
            exp.select(exp.Literal.number(1))
            .from_(exp.alias_(resolve(table.key), "x"))
            .where(
                exp.And(
                    this=exp.EQ(
                        this=_col("x", edge.from_columns[0]), expression=_col("l", "entity_id")
                    ),
                    expression=exp.LT(
                        this=_time_expr(compiled.dialect, "x", table), expression=cutoff_time
                    ),
                )
            )
        )
        case = exp.Case(
            ifs=[exp.If(this=exp.Exists(this=inner), true=exp.Literal.number(1))],
            default=exp.Literal.number(0),
        )
        selects.append(exp.alias_(exp.Sum(this=case), f"covered_{len(names)}"))
        names.append(table.name)
    if not names:
        return None
    sub = exp.alias_(exp.Subquery(this=compiled.select.copy()), "l")
    query = (
        exp.select(*selects)
        .from_(sub)
        .where(exp.EQ(this=cutoff_time, expression=_stamp(compiled.dialect, first, tz=False)))
    )
    return query, names


def _read_coverage(
    spec: TaskSpec,
    graph: SchemaGraph,
    compiled: CompiledLabels,
    resolve: TableResolver,
    query: Callable[[str], Any],
    held: dict[str, list[str]] | None = None,
) -> list[TableCoverage]:
    built = coverage_select(spec, graph, compiled, resolve, held)
    if built is None:
        return []
    select, names = built
    frame = arrow_to_pandas(query(select.sql(compiled.dialect)))
    row = frame.iloc[0]
    eligible = int(row["eligible"])
    return [
        TableCoverage(name, eligible, int(row[f"covered_{i}"]) if eligible else 0)
        for i, name in enumerate(names)
    ]


# -- running the labels -----------------------------------------------------------------------

COUNT_LIMIT = 10_000  # one row per cutoff, at most MAX_CUTOFFS
QUERY_TIMEOUT_S = 300


@dataclass(frozen=True)
class CutoffCount:
    cutoff: datetime
    window_end: datetime
    eligible: int
    positives: int | None  # binary tasks
    base_rate: float | None
    mean_label: float | None  # regression tasks


@dataclass(frozen=True)
class TableCoverage:
    """How many of the entities at the first cutoff have any row in a related table before it."""

    table: str
    eligible: int
    covered: int

    @property
    def share(self) -> float:
        return self.covered / self.eligible if self.eligible else 0.0


@dataclass(frozen=True)
class LabelRun:
    compiled: CompiledLabels
    dialect: str
    counts: list[CutoffCount]
    table: TableRef | None  # where the labels were materialized, if they were
    coverage: list[TableCoverage] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(c.eligible for c in self.counts)


def _read_counts(
    compiled: CompiledLabels, binary: bool, query: Callable[[str], Any]
) -> list[CutoffCount]:
    frame = arrow_to_pandas(
        query(counts_select(compiled.select, binary=binary).sql(compiled.dialect))
    )
    windows = {w.cutoff.replace(tzinfo=None): w for w in compiled.kept}
    out: list[CutoffCount] = []
    for r in frame.itertuples():
        window = windows[r.cutoff_time.to_pydatetime().replace(tzinfo=None)]
        eligible = int(r.eligible)
        if binary:
            positives = int(r.positives)
            out.append(
                CutoffCount(
                    window.cutoff,
                    window.window_end,
                    eligible,
                    positives,
                    positives / eligible if eligible else None,
                    None,
                )
            )
        else:
            out.append(
                CutoffCount(
                    window.cutoff,
                    window.window_end,
                    eligible,
                    None,
                    None,
                    None if r.mean_label is None else float(r.mean_label),
                )
            )
    return out


def missing_columns(spec: TaskSpec, graph: SchemaGraph, description: DataDescription) -> list[str]:
    """``table.column`` the labels read that a snapshot does not hold."""
    missing: list[str] = []
    for key, columns in required_columns(spec, graph).items():
        held = description.tables.get(key)
        if held is None:
            missing.extend(f"{key}.{c}" for c in columns)
        else:
            missing.extend(f"{key}.{c}" for c in columns if c not in held.columns)
    return missing


def label_table_ref(task_id: str, version_id: str) -> TableRef:
    """``mlpilot.labels_<task>_<data version>``: the same task on the same data is the same table."""
    return TableRef(f"labels_{task_id.replace('-', '')[:12]}_{version_id[:8]}", INTERNAL_SCHEMA)


def run_on_snapshot(
    spec: TaskSpec,
    graph: SchemaGraph,
    workspace: Workspace,
    version_id: str,
    description: DataDescription,
    *,
    task_id: str,
    materialize: bool = True,
) -> LabelRun:
    """Compile the labels for a database snapshot and run them in the project's DuckDB file."""
    missing = missing_columns(spec, graph, description)
    if missing:
        raise LabelError(
            "The snapshot does not hold what the labels read: "
            + ", ".join(missing)
            + ". Take a new snapshot that includes them"
        )
    refs = workspace.snapshot_tables(version_id)

    def resolve(key: str) -> exp.Table:
        ref = refs[key]
        return exp.table_(ref.name, db=ref.schema, quoted=True)

    compiled = compile_labels(spec, graph, "duckdb", resolve, description.as_of)

    def query(sql: str) -> Any:
        return workspace.source.query(sql, limit=COUNT_LIMIT, timeout_s=QUERY_TIMEOUT_S)

    counts = _read_counts(compiled, spec.target.type == "binary", query)
    table: TableRef | None = None
    if materialize:
        table = label_table_ref(task_id, version_id)
        # MLPilot's own statement over MLPilot's own tables; the DuckDB file has no external access.
        workspace.source.execute(
            f"CREATE OR REPLACE TABLE {quote_ident(table.schema)}.{quote_ident(table.name)} AS "
            f"{compiled.sql}"
        )
    held = {key: list(t.columns) for key, t in description.tables.items()}
    coverage = _read_coverage(spec, graph, compiled, resolve, query, held)
    return LabelRun(compiled, "duckdb", counts, table, coverage)


def run_on_source(
    spec: TaskSpec, graph: SchemaGraph, source: SourceWithChecks, as_of: datetime
) -> LabelRun:
    """Compile the labels for the live database and run them read-only through the SQL guard."""
    tables = {t.key: t for t in graph.tables}

    def resolve(key: str) -> exp.Table:
        t = tables[key]
        return exp.table_(t.name, db=t.db_schema or None, quoted=True)

    compiled = compile_labels(spec, graph, source.dialect, resolve, as_of)
    allowed = sql_guard.allowed_tables(source)

    def query(sql: str) -> Any:
        q = sql_guard.guard(sql, source.dialect, allowed_tables=allowed)
        return sql_guard.execute(source, q, limit=COUNT_LIMIT, timeout_s=QUERY_TIMEOUT_S)

    return LabelRun(
        compiled,
        source.dialect,
        _read_counts(compiled, spec.target.type == "binary", query),
        None,
        _read_coverage(spec, graph, compiled, resolve, query),
    )
