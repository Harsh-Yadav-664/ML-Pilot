"""Baseline features without an LLM (#55): Deep Feature Synthesis over the schema graph.

For every table reachable from the entity table by one or two edges in the child direction
(customers -> orders -> order_items), ``generate`` writes feature IRs from a fixed set of
templates and compiles them with ``ml.features.compile``, so a baseline feature is the same
kind of object as any other feature: readable SQL, bounded by the cutoff, checked by the
point-in-time guard. There is no second code path.

Templates, in priority order (the cap keeps the first ones):

0. counts over 7, 30, 90, 365 days and over all history
1. days since the most recent and the first event
2. counts per top category value; recent share (count in 7 days / 30 days, 30 / 90 days)
3. sum and mean of numeric columns, share of rows where a boolean column is true
4. min, max and standard deviation of numeric columns
5. attributes of the entity row itself, and its age at the cutoff

A table is skipped, with the reason recorded, if it has no event time and was not confirmed as
static: its rows could not be placed before the cutoff.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from ml.data.schema_graph import Edge, EdgeRef, SchemaGraph, Table, type_kind
from ml.features.compile import compile, compile_entity_value
from ml.features.ir import FeatureIR, Predicate, describe

Group = Literal["count", "recency", "category", "trend", "numeric", "boolean", "attribute"]

WINDOWS = (7, 30, 90, 365)
CATEGORY_WINDOWS = (30, 90, 365)
BOOLEAN_WINDOWS = (30, 90, None)
NUMERIC_WINDOWS = (30, 90, 365)
TOP_CATEGORIES = 5
MAX_DEPTH = 2
DEFAULT_MAX_FEATURES = 300
MAX_NAME = 63


@dataclass
class Candidate:
    name: str
    group: Group
    priority: int
    description: str
    sql: str
    ir: FeatureIR | None = None  # None for attributes of the entity row
    tables: list[str] = field(default_factory=list)  # the path, entity first

    @property
    def ir_json(self) -> dict[str, Any] | None:
        return self.ir.model_dump(mode="json") if self.ir else None


@dataclass
class Skipped:
    table: str
    reason: str


@dataclass
class Generated:
    candidates: list[Candidate]
    skipped: list[Skipped]


def _short(name: str) -> str:
    if len(name) <= MAX_NAME:
        return name
    return name[: MAX_NAME - 9] + "_" + hashlib.sha1(name.encode()).hexdigest()[:8]


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in text.lower())
    return "_".join(p for p in out.split("_") if p) or "x"


def _children(graph: SchemaGraph, table: str) -> list[Edge]:
    """Edges whose parent is ``table``: the child tables, in a stable order."""
    return sorted(
        (
            e
            for e in graph.edges
            if e.to_table == table and e.from_table != table and len(e.from_columns) == 1
        ),
        key=lambda e: (e.from_table, e.from_columns),
    )


def _ref(e: Edge) -> EdgeRef:
    return EdgeRef(
        from_table=e.from_table,
        from_columns=e.from_columns,
        to_table=e.to_table,
        to_columns=e.to_columns,
    )


def _usable(table: Table) -> str | None:
    if table.is_static is True:
        return None
    if table.time_column and table.time_leakage_hint and table.time_column_source != "user":
        return f"its only time column, {table.time_column}, is a last-modified time"
    if table.time_column:
        return None
    return "no event time and not confirmed as static"


def _paths(graph: SchemaGraph, entity: str) -> tuple[list[list[Edge]], list[Skipped]]:
    tables = {t.key: t for t in graph.tables}
    found: list[list[Edge]] = []
    skipped: dict[str, str] = {}

    def walk(here: str, trail: list[Edge], seen: set[str]) -> None:
        for edge in _children(graph, here):
            child = edge.from_table
            if child in seen:
                continue
            why = _usable(tables[child])
            if why:
                skipped[child] = why
                continue
            path = [*trail, edge]
            found.append(path)
            if len(path) < MAX_DEPTH:
                walk(child, path, {*seen, child})

    walk(entity, [], {entity})
    return found, [Skipped(t, r) for t, r in sorted(skipped.items())]


def _has_event(path: list[Edge], tables: dict[str, Table]) -> bool:
    return any(tables[e.from_table].time_column for e in path)


def generate(
    graph: SchemaGraph,
    entity_table: str,
    *,
    entity_created_at: str | None = None,
    top_values: dict[tuple[str, str], list[str]] | None = None,
    dialect: str = "duckdb",
) -> Generated:
    """Every baseline feature candidate for ``entity_table``, compiled and guard-checked, best first.

    ``top_values`` maps ``(table, column)`` to the most frequent values of a text column; without
    it no per-category counts are made. They go into SQL only, never into a prompt.
    """
    tables = {t.key: t for t in graph.tables}
    if entity_table not in tables:
        raise ValueError(f"{entity_table!r} is not a table of the schema")
    top = top_values or {}
    paths, skipped = _paths(graph, entity_table)
    out: list[Candidate] = []

    def add(
        ir: FeatureIR, group: Group, priority: int, path: list[Edge], depth_names: list[str]
    ) -> None:
        sql = compile(ir, graph, dialect)  # raises if the IR or the guard says no: a bug here
        out.append(
            Candidate(
                name=ir.name,
                group=group,
                priority=priority,
                description=describe(ir),
                sql=sql,
                ir=ir,
                tables=[entity_table, *depth_names],
            )
        )

    for path in paths:
        source = tables[path[-1].from_table]
        names = [e.from_table for e in path]
        stem = "__".join(_slug(n.split(".")[-1]) for n in names)
        has_event = _has_event(path, tables)

        def make(
            tag: str,
            agg: str,
            *,
            column: str | None = None,
            window: int | None = None,
            filters: list[Predicate] | None = None,
            ratio_to: FeatureIR | None = None,
            _path: list[Edge] = path,
            _source: Table = source,
            _stem: str = stem,
        ) -> FeatureIR:
            suffix = f"_{window}d" if window else "_all"
            return FeatureIR(
                name=_short(f"{_stem}__{tag}{suffix}"),
                entity_table=entity_table,
                path=[_ref(e) for e in _path],
                source_table=_source.key,
                agg=agg,  # type: ignore[arg-type]
                column=column,
                window_days=window,
                filter=filters or [],
                ratio_to=ratio_to,
            )

        windows: tuple[int | None, ...] = (*WINDOWS, None) if has_event else (None,)
        for w in windows:
            add(make("count", "count", window=w), "count", 0, path, names)
        if has_event:
            add(make("days_since_last", "days_since_last"), "recency", 1, path, names)
            add(make("days_since_first", "days_since_first"), "recency", 1, path, names)
            for short, long in ((7, 30), (30, 90)):
                numerator = make("count", "count", window=short)
                denominator = make("count", "count", window=long)
                ir = numerator.model_copy(
                    update={
                        "name": _short(f"{stem}__count_{short}d_over_{long}d"),
                        "ratio_to": denominator,
                    }
                )
                add(ir, "trend", 2, path, names)
        for column in source.columns:
            kind = type_kind(column.type)
            if column.hint in ("id", "time") or column.name in source.primary_key:
                continue
            if kind == "text" and (source.key, column.name) in top:
                for value in top[(source.key, column.name)][:TOP_CATEGORIES]:
                    flt = [Predicate(column=column.name, op="=", value=value)]
                    tag = f"count_{_slug(column.name)}_{_slug(str(value))}"
                    for w in CATEGORY_WINDOWS if has_event else (None,):
                        add(make(tag, "count", window=w, filters=flt), "category", 2, path, names)
            elif kind == "boolean":
                flt = [Predicate(column=column.name, op="=", value=True)]
                for w in BOOLEAN_WINDOWS if has_event else (None,):
                    ir = make(f"share_{_slug(column.name)}", "share", window=w, filters=flt)
                    add(ir, "boolean", 3, path, names)
            elif kind in ("integer", "float") and column.hint == "numeric":
                for agg, priority in (
                    ("sum", 3),
                    ("mean", 3),
                    ("min", 4),
                    ("max", 4),
                    ("std", 4),
                ):
                    for w in NUMERIC_WINDOWS if has_event else (None,):
                        ir = make(f"{agg}_{_slug(column.name)}", agg, column=column.name, window=w)
                        add(ir, "numeric", priority, path, names)

    skipped = [
        *skipped,
        *_attributes(graph, tables[entity_table], entity_created_at, top, dialect, out),
    ]
    out.sort(key=lambda c: (c.priority, len(c.tables), c.name))
    seen: set[str] = set()
    for c in out:
        if c.name in seen:
            raise ValueError(f"two baseline features are both called {c.name!r}")
        seen.add(c.name)
    return Generated(out, skipped)


def _attributes(
    graph: SchemaGraph,
    entity: Table,
    created_at: str | None,
    top: dict[tuple[str, str], list[str]],
    dialect: str,
    out: list[Candidate],
) -> list[Skipped]:
    why = _usable(entity)
    if why:
        return [Skipped(entity.key, f"entity attributes: {why}")]
    stem = _slug(entity.name)
    for column in entity.columns:
        kind = type_kind(column.type)
        if column.name in entity.primary_key or column.hint in ("id", "time"):
            continue
        if kind not in ("integer", "float", "boolean", "text"):
            continue
        if kind == "text" and (entity.key, column.name) not in top:
            continue  # free text and identifiers such as e-mail addresses are not categories
        sql = compile_entity_value(graph, entity.key, column.name, dialect)
        out.append(
            Candidate(
                name=_short(f"{stem}__{_slug(column.name)}"),
                group="attribute",
                priority=5,
                description=f"Column {column.name} of {entity.name}",
                sql=sql,
                tables=[entity.key],
            )
        )
    start = created_at or entity.time_column
    if start:
        out.append(
            Candidate(
                name=_short(f"{stem}__age_days"),
                group="attribute",
                priority=5,
                description=f"Days between {start.replace('_', ' ')} and the cutoff",
                sql=compile_entity_value(graph, entity.key, None, dialect, created_at=start),
                tables=[entity.key],
            )
        )
    return []
