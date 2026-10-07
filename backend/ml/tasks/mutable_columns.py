"""Find columns whose value is overwritten after its row exists (#139).

The point-in-time guard bounds rows by their event time, which does not help for a column that is
written or changed later: ``customers.is_churned`` is set when a customer leaves, ``orders.status``
moves from 'paid' to 'refunded'. Reading either from a row older than the cutoff shows what
happened afterwards. A name check finds the obvious ones (``schema_graph.mutable_name_reason``).
This module finds the others from data: two snapshots of the same database taken at different
times, compared for the rows that are in both. A column whose value differs for a row that
already existed is mutable, whatever it is called.

It needs two snapshots (#97), and it can only see changes that happened between them: a column
that did not change in that time is not proven immutable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from ml.data.schema_graph import SchemaGraph


@dataclass(frozen=True)
class Observation:
    table: str
    column: str
    compared_rows: int
    changed_rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "column": self.column,
            "compared_rows": self.compared_rows,
            "changed_rows": self.changed_rows,
        }


@dataclass
class MutableReport:
    mutable: list[Observation] = field(default_factory=list)
    checked_tables: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (table, why)

    def as_overrides(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for o in self.mutable:
            out.setdefault(o.table, []).append(o.column)
        return out


def _differs(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise difference where two missing values are equal."""
    return ~((a == b) | (a.isna() & b.isna()))


def observe_mutable(
    older: Mapping[str, pd.DataFrame], newer: Mapping[str, pd.DataFrame], graph: SchemaGraph
) -> MutableReport:
    """Compare the two snapshots table by table (keys are the graph's table keys).

    A table is compared on its primary key; one without a key, or missing from a snapshot, is
    skipped and listed, never silently ignored."""
    report = MutableReport()
    for table in graph.tables:
        if table.key not in older or table.key not in newer:
            report.skipped.append((table.key, "not in both snapshots"))
            continue
        if not table.primary_key:
            report.skipped.append((table.key, "no primary key to match rows by"))
            continue
        keys = list(table.primary_key)
        old, new = older[table.key], newer[table.key]
        if old.duplicated(keys).any() or new.duplicated(keys).any():
            report.skipped.append((table.key, "its primary key is not unique in the data"))
            continue
        both = old.merge(new, on=keys, how="inner", suffixes=("__old", "__new"))
        report.checked_tables += 1
        for column in table.columns:
            name = column.name
            if name in keys or f"{name}__old" not in both or f"{name}__new" not in both:
                continue
            changed = int(_differs(both[f"{name}__old"], both[f"{name}__new"]).sum())
            if changed:
                report.mutable.append(Observation(table.key, name, len(both), changed))
    return report
