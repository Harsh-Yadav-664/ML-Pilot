"""Feature IR (#100): one small typed shape that covers most useful relational features.

A feature is "aggregate a column of a related table, over a window before the cutoff, with a
filter". ``FeatureIR`` is that sentence as data. A cheap model can fill it in, a person can read
it, and ``ml.features.compile`` turns it into SQL that is point-in-time safe by construction.

This module holds the model, the validation against the schema graph (every problem comes back
as a field-level error, so a repair prompt can name the field) and ``describe``, the English
sentence used in narration and in the evidence report.

The IR is data, never code: identifiers must exist in the schema graph and literals are typed,
so nothing a model writes here can reach the SQL as text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from ml.data.schema_graph import EdgeRef, SchemaGraph, Table, type_kind

Agg = Literal[
    "count",
    "count_distinct",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "days_since_last",
    "days_since_first",
    "share",
]
Op = Literal["=", "!=", "<", "<=", ">", ">=", "in", "is_null", "is_not_null"]
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr

NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
MAX_WINDOW_DAYS = 3650
MAX_FILTERS = 5
MAX_IN_VALUES = 50

NUMERIC_AGGS = ("sum", "mean", "min", "max", "std")
NO_COLUMN_AGGS = ("count", "days_since_last", "days_since_first", "share")
# log1p needs a value that cannot be negative; the other aggregates of a numeric column can be
NON_NEGATIVE_AGGS = ("count", "count_distinct", "std", "days_since_last", "days_since_first")


class Predicate(BaseModel):
    """``column op value`` on the source table. SQL semantics: NULL never matches ``!=`` or ``<``."""

    column: str
    op: Op
    value: Scalar | list[Scalar] | None = None


class FeatureIR(BaseModel):
    name: str = Field(description="snake_case feature name")
    entity_table: str = Field(
        description="The table the label rows are about (the path starts here)"
    )
    path: list[EdgeRef] = Field(
        min_length=1, description="Schema-graph edges from the entity table to the source table"
    )
    source_table: str
    agg: Agg
    column: str | None = Field(
        None, description="None for count, share and days_since_*; required otherwise"
    )
    window_days: int | None = Field(
        None, description="Rows from the N days before the cutoff; None = all history"
    )
    filter: list[Predicate] = Field(default_factory=list)
    ratio_to: FeatureIR | None = Field(
        None,
        description="Divide by this feature (one level only), e.g. 30-day count / 90-day count",
    )
    transform: Literal["none", "log1p"] = "none"


@dataclass(frozen=True)
class FieldError:
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}


class FeatureIRError(ValueError):
    """The IR does not fit the schema; ``errors`` names every offending field."""

    def __init__(self, errors: list[FieldError]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e.field}: {e.message}" for e in errors))


@dataclass(frozen=True)
class Hop:
    """One step of the path, oriented from the table we are at to the next one."""

    table: Table
    here_column: str
    there_column: str


@dataclass(frozen=True)
class Resolved:
    """What validation worked out; the compiler builds on it."""

    entity: Table
    hops: list[Hop]  # one per path edge; hops[-1].table is the source table
    event: Hop | None  # the hop whose table supplies the event time (nearest to the source)


_Bad = Callable[[str, str], None]


# -- validation --------------------------------------------------------------------------------


def validate(ir: FeatureIR, graph: SchemaGraph) -> list[FieldError]:
    return _check(ir, graph, "", nested=False)[0]


def resolve(ir: FeatureIR, graph: SchemaGraph) -> Resolved:
    errors, resolved = _check(ir, graph, "", nested=False)
    if errors or resolved is None:
        raise FeatureIRError(errors)
    return resolved


def _check(
    ir: FeatureIR, graph: SchemaGraph, prefix: str, *, nested: bool
) -> tuple[list[FieldError], Resolved | None]:
    errors: list[FieldError] = []

    def bad(field: str, message: str) -> None:
        errors.append(FieldError(prefix + field, message))

    if not NAME.match(ir.name):
        bad("name", "use lower-case letters, digits and underscores, starting with a letter")
    tables = {t.key: t for t in graph.tables}
    entity = tables.get(ir.entity_table)
    if entity is None:
        bad("entity_table", f"{ir.entity_table!r} is not a table of the schema")
    hops = _walk(ir, graph, tables, entity, bad)
    if hops is None or entity is None:
        return errors, None
    source = hops[-1].table
    event = next((h for h in reversed(hops) if h.table.time_column), None)

    for i, hop in enumerate(hops):
        if not hop.table.time_column and hop.table.is_static is not True:
            bad(
                f"path[{i}]",
                f"table {hop.table.key!r} has no event time and is not confirmed as static; "
                "its rows cannot be placed before the cutoff",
            )
    _check_agg(ir, source, event, bad)
    _check_window(ir, event, bad)
    _check_filters(ir, source, bad)
    if ir.transform == "log1p" and ir.agg not in NON_NEGATIVE_AGGS:
        bad(
            "transform",
            f"log1p needs a value that cannot be negative; {ir.agg!r} of a column can be, "
            f"use one of {', '.join(NON_NEGATIVE_AGGS)}",
        )
    if ir.ratio_to is not None:
        if nested:
            bad("ratio_to", "ratio_to can be used one level deep only")
        else:
            inner, _ = _check(ir.ratio_to, graph, prefix + "ratio_to.", nested=True)
            errors.extend(inner)
            if ir.ratio_to.entity_table != ir.entity_table:
                bad("ratio_to.entity_table", "must be the same entity table as the feature")
            if ir.ratio_to.transform != "none":
                bad("ratio_to.transform", "apply the transform to the outer feature, not here")
    if errors:
        return errors, None
    return errors, Resolved(entity=entity, hops=hops, event=event)


def _walk(
    ir: FeatureIR,
    graph: SchemaGraph,
    tables: dict[str, Table],
    entity: Table | None,
    bad: _Bad,
) -> list[Hop] | None:
    if entity is None:
        return None
    hops: list[Hop] = []
    here = entity
    seen = {entity.key}
    ok = True
    for i, step in enumerate(ir.path):
        field = f"path[{i}]"
        edge = next((e for e in graph.edges if step.matches(e)), None)
        if edge is None:
            bad(field, f"no such edge in the schema graph: {_edge_text(step)}")
            return None
        if len(edge.from_columns) != 1:
            bad(field, "edges on composite keys are not supported")
            return None
        if here.key == edge.from_table:
            there, here_col, there_col = (
                tables[edge.to_table],
                edge.from_columns[0],
                edge.to_columns[0],
            )
        elif here.key == edge.to_table:
            there, here_col, there_col = (
                tables[edge.from_table],
                edge.to_columns[0],
                edge.from_columns[0],
            )
        else:
            bad(field, f"the edge {_edge_text(step)} does not start at {here.key!r}")
            return None
        if there.key in seen:
            bad(field, f"the path comes back to {there.key!r}")
            return None
        if i == 0 and entity.primary_key != [here_col]:
            bad(
                field,
                f"the first edge must leave the entity table through its key "
                f"{entity.primary_key}, not {here_col!r}",
            )
            ok = False
        seen.add(there.key)
        hops.append(Hop(table=there, here_column=here_col, there_column=there_col))
        here = there
    if here.key != ir.source_table:
        bad("source_table", f"the path ends at {here.key!r}, not at {ir.source_table!r}")
        ok = False
    return hops if ok else None


def _check_agg(
    ir: FeatureIR,
    source: Table,
    event: Hop | None,
    bad: _Bad,
) -> None:
    if ir.agg in NO_COLUMN_AGGS:
        if ir.column is not None:
            bad("column", f"{ir.agg!r} does not take a column; leave it empty")
    elif ir.column is None:
        bad("column", f"{ir.agg!r} needs a column of {source.key!r}")
    else:
        column = next((c for c in source.columns if c.name == ir.column), None)
        if column is None:
            bad("column", f"{source.key!r} has no column {ir.column!r}")
        else:
            kind = type_kind(column.type)
            if ir.agg in NUMERIC_AGGS and kind not in ("integer", "float"):
                bad(
                    "column",
                    f"{ir.agg!r} needs a numeric column, {ir.column!r} is {column.type}",
                )
    if ir.agg in ("days_since_last", "days_since_first") and event is None:
        bad("agg", f"{ir.agg!r} needs a table with an event time on the path")
    if ir.agg == "share" and not ir.filter:
        bad("filter", "'share' is the share of rows matching the filter; give at least one")


def _check_window(ir: FeatureIR, event: Hop | None, bad: _Bad) -> None:
    if ir.window_days is None:
        return
    if not 1 <= ir.window_days <= MAX_WINDOW_DAYS:
        bad("window_days", f"must be between 1 and {MAX_WINDOW_DAYS} days")
    if event is None:
        bad("window_days", "a window needs a table with an event time on the path")


def _check_filters(ir: FeatureIR, table: Table, bad: _Bad) -> None:
    if len(ir.filter) > MAX_FILTERS:
        bad("filter", f"at most {MAX_FILTERS} predicates")
    source = {c.name: c for c in table.columns}
    for i, pred in enumerate(ir.filter[:MAX_FILTERS]):
        field = f"filter[{i}]"
        column = source.get(pred.column)
        if column is None:
            bad(f"{field}.column", f"{ir.source_table!r} has no column {pred.column!r}")
            continue
        kind = type_kind(column.type)
        if pred.op in ("is_null", "is_not_null"):
            if pred.value is not None:
                bad(f"{field}.value", f"{pred.op!r} takes no value")
            continue
        if pred.op == "in":
            if not isinstance(pred.value, list) or not pred.value:
                bad(f"{field}.value", "'in' needs a non-empty list")
                continue
            if len(pred.value) > MAX_IN_VALUES:
                bad(f"{field}.value", f"at most {MAX_IN_VALUES} values")
                continue
            values: list[Scalar] = list(pred.value)
        else:
            if pred.value is None or isinstance(pred.value, list):
                bad(f"{field}.value", f"{pred.op!r} needs a single value")
                continue
            values = [pred.value]
        if kind == "boolean" and pred.op not in ("=", "!="):
            bad(f"{field}.op", "a boolean column only supports = and !=")
            continue
        if kind == "text" and pred.op in ("<", "<=", ">", ">="):
            bad(f"{field}.op", "ordering a text column is not supported; use =, != or in")
            continue
        for v in values:
            message = _value_problem(kind, v, column.name)
            if message:
                bad(f"{field}.value", message)
                break


def _value_problem(kind: str, value: Scalar, column: str) -> str | None:
    if kind in ("integer", "float"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"{column!r} is numeric; give a number, not {value!r}"
    elif kind == "boolean":
        if not isinstance(value, bool):
            return f"{column!r} is boolean; give true or false"
    elif kind == "text":
        if not isinstance(value, str):
            return f"{column!r} is text; give a string"
    elif kind in ("date", "timestamp"):
        if parse_time(value) is None:
            return f"{column!r} is a {kind}; give an ISO date or timestamp such as '2024-01-31'"
    else:
        return f"filtering on {column!r} (type {kind}) is not supported"
    return None


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(value), datetime.min.time())
        except ValueError:
            return None


def _edge_text(step: EdgeRef) -> str:
    return (
        f"{step.from_table}({', '.join(step.from_columns)}) -> "
        f"{step.to_table}({', '.join(step.to_columns)})"
    )


# -- describe ----------------------------------------------------------------------------------

_OPS = {
    "=": "is",
    "!=": "is not",
    "<": "is below",
    "<=": "is at most",
    ">": "is above",
    ">=": "is at least",
    "in": "is one of",
}


def _words(name: str) -> str:
    return name.replace("_", " ")


def _literal(value: Scalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, str) else str(value)


def _condition(pred: Predicate) -> str:
    col = _words(pred.column)
    if pred.op == "is_null":
        return f"{col} is missing"
    if pred.op == "is_not_null":
        return f"{col} is present"
    if pred.op == "in" and isinstance(pred.value, list):
        return f"{col} is one of {', '.join(_literal(v) for v in pred.value)}"
    assert pred.value is not None and not isinstance(pred.value, list)
    return f"{col} {_OPS[pred.op]} {_literal(pred.value)}"


def _via(ir: FeatureIR) -> list[str]:
    """The tables between the entity and the source, in order."""
    here, out = ir.entity_table, []
    for step in ir.path[:-1]:
        here = step.to_table if step.from_table == here else step.from_table
        out.append(here)
    return out


def describe(ir: FeatureIR) -> str:
    """The feature as one English sentence, for narration and the evidence report."""
    rows = _words(ir.source_table)
    column = _words(ir.column) if ir.column else ""
    if ir.agg == "count":
        head = f"Number of {rows}"
    elif ir.agg == "count_distinct":
        head = f"Number of distinct {column} values among {rows}"
    elif ir.agg == "sum":
        head = f"Total {column} of {rows}"
    elif ir.agg == "mean":
        head = f"Average {column} of {rows}"
    elif ir.agg == "min":
        head = f"Smallest {column} among {rows}"
    elif ir.agg == "max":
        head = f"Largest {column} among {rows}"
    elif ir.agg == "std":
        head = f"Standard deviation of {column} across {rows}"
    elif ir.agg == "days_since_last":
        head = f"Days since the most recent of the {rows}"
    elif ir.agg == "days_since_first":
        head = f"Days since the earliest of the {rows}"
    else:
        head = f"Share of {rows}"
    if ir.filter:
        head += " where " + " and ".join(_condition(p) for p in ir.filter)
    if len(ir.path) > 1:
        head += ", reached through " + " and ".join(_words(t) for t in _via(ir)) + ","
    if ir.window_days:
        head += f" in the {ir.window_days} days before the cutoff"
    else:
        head += " before the cutoff"
    if ir.ratio_to is not None:
        other = describe(ir.ratio_to)
        head = f"{head}, divided by: {other[0].lower()}{other[1:]}"
    if ir.transform == "log1p":
        head = f"Log of (1 + value): {head[0].lower()}{head[1:]}"
    return head
