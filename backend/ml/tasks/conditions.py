"""The small condition language of task specs: a `WHERE`-style test on one table's columns.

A condition is a text such as ``status != 'cancelled' AND total > 10`` or
``ordered_at >= :cutoff - interval '90 days'``. It is parsed with sqlglot and checked against a
whitelist of node types, so a condition can compare columns of its own table with literals and
with ``:cutoff`` (plus or minus an interval), combined with AND, OR and NOT. No function, cast,
subquery, join, other placeholder or statement gets through. The label compiler (#50) renders the
checked tree, which is why this is a language and not free SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

CUTOFF = "cutoff"

_ALLOWED: tuple[type[exp.Expr], ...] = (
    exp.Column,
    exp.Identifier,
    exp.Literal,
    exp.Null,
    exp.Boolean,
    exp.Placeholder,
    exp.Interval,
    exp.Var,
    exp.Paren,
    exp.Tuple,
    exp.And,
    exp.Or,
    exp.Not,
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.In,
    exp.Is,
    exp.Like,
    exp.ILike,
    exp.Between,
    exp.Add,
    exp.Sub,
    exp.Neg,
)


_INTERVAL = re.compile(r"^(\d+)\s*(second|minute|hour|day|week|month|year)s?$", re.IGNORECASE)


class ConditionError(ValueError):
    """The condition is not valid in the task-spec condition language."""


@dataclass(frozen=True)
class Condition:
    text: str
    tree: exp.Expr
    columns: tuple[str, ...]
    uses_cutoff: bool


def parse_condition(text: str, *, table: str | None = None) -> Condition:
    """Parse and whitelist-check ``text``; ``table`` allows ``table.column`` references to it."""
    if not isinstance(text, str) or not text.strip():
        raise ConditionError("the condition is empty")
    try:
        trees = sqlglot.parse(text, dialect="postgres")
    except SqlglotError as e:
        raise ConditionError(f"cannot parse the condition: {str(e).splitlines()[0]}") from None
    if len(trees) != 1 or trees[0] is None:
        raise ConditionError("a condition is a single expression, not several statements")
    tree = trees[0]
    columns: list[str] = []
    uses_cutoff = False
    for node in tree.walk():
        if not isinstance(node, _ALLOWED):
            raise ConditionError(
                f"{_describe(node)} is not allowed in a condition. Use column comparisons with "
                "AND, OR, NOT, IN, IS NULL, LIKE, BETWEEN, literals and :cutoff"
            )
        if isinstance(node, exp.Interval):
            unit = node.args.get("unit")
            amount = f"{node.this.name} {unit.name}" if unit is not None else node.this.name
            if not _INTERVAL.match(amount.strip()):
                raise ConditionError(
                    f"interval {amount!r} is not a whole number of seconds, minutes, hours, days, "
                    "weeks, months or years"
                )
        if isinstance(node, exp.Placeholder):
            if str(node.this).lower() != CUTOFF:
                raise ConditionError(f"unknown placeholder :{node.this}; only :{CUTOFF} exists")
            uses_cutoff = True
        elif isinstance(node, exp.Column):
            qualifier = node.table
            if qualifier and (table is None or qualifier.lower() != table.lower()):
                raise ConditionError(
                    f"column {node.sql(dialect='postgres')} refers to another table; a condition "
                    "may only use columns of its own table"
                )
            if node.name not in columns:
                columns.append(node.name)
    return Condition(text=text.strip(), tree=tree, columns=tuple(columns), uses_cutoff=uses_cutoff)


def _describe(node: exp.Expr) -> str:
    if isinstance(node, exp.Subquery | exp.Select):
        return "a subquery"
    if isinstance(node, exp.Func):
        return f"the function {node.sql_name()}"
    return f"{type(node).__name__}"
