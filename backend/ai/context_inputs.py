"""Turn MLPilot's own profiles into `DatasetContext` for the context builder.

Both inputs hold more than a prompt may contain (category labels, top values). The builder decides
what goes in; these functions only translate names and keep the labels available for the levels
that allow them.
"""

from __future__ import annotations

from typing import Any

from ai.context_builder import ColumnContext, DatasetContext, Relationship, TableContext
from ml.core.interfaces import ProfileResult
from ml.data.profiling.db_stats import TableStats
from ml.data.schema_graph import SchemaGraph

# CSV profile key -> prompt statistic (ml/data/profiling/stats.py)
_PROFILE_KEYS = {
    "count": "non_null",
    "missing_pct": "null_fraction",
    "n_unique": "distinct",
    "mean": "mean",
    "std": "stddev",
    "min": "min",
    "max": "max",
    "median": "p50",
    "skew": "skew",
}


def dataset_from_profile(
    profile: ProfileResult,
    target_column: str | None = None,
    table_name: str = "dataset",
    *,
    skip_target: bool = False,
) -> DatasetContext:
    """A one-table dataset from the profile of an uploaded CSV."""
    columns = []
    for name, raw in profile.column_stats.items():
        if skip_target and name == target_column:
            continue
        stats = {out: raw[key] for key, out in _PROFILE_KEYS.items() if raw.get(key) is not None}
        top = raw.get("top_values") or {}
        columns.append(
            ColumnContext(
                name=name,
                type=str(raw.get("dtype") or raw.get("type") or ""),
                hint="target" if name == target_column else None,
                stats=stats,
                category_labels=[(str(k), int(v)) for k, v in top.items()] or None,
            )
        )
    return DatasetContext(
        tables=[TableContext(name=table_name, columns=columns, row_count=profile.rows)],
        notes=list(profile.warnings),
    )


def dataset_from_graph(
    graph: SchemaGraph, stats: dict[str, TableStats] | None = None
) -> DatasetContext:
    """A multi-table dataset from the schema graph of a connected database and, optionally, its column statistics."""
    tables = []
    for table in graph.tables:
        table_stats = (stats or {}).get(table.key)
        by_name = {c.name: c for c in table_stats.columns} if table_stats else {}
        columns = []
        for col in table.columns:
            s = by_name.get(col.name)
            values: dict[str, Any] = {}
            labels = None
            if s is not None:
                values = {
                    "non_null": s.non_null,
                    "null_fraction": s.null_fraction,
                    "distinct": s.distinct,
                    "mean": s.mean,
                    "stddev": s.stddev,
                    "min": s.min,
                    "max": s.max,
                    "p1": s.p1,
                    "p50": s.p50,
                    "p99": s.p99,
                    "time_min": s.time_min,
                    "time_max": s.time_max,
                }
                labels = [(t.value, t.count) for t in s.top_values] or None
            columns.append(
                ColumnContext(
                    name=col.name,
                    type=col.type,
                    hint=col.hint,
                    primary_key=col.is_primary_key,
                    stats=values,
                    category_labels=labels,
                )
            )
        tables.append(
            TableContext(
                name=table.key,
                columns=columns,
                row_count=table.row_count,
                row_count_estimated=table.row_count_estimated,
                time_column=table.time_column,
                notes=[table.time_leakage_hint] if table.time_leakage_hint else [],
            )
        )
    return DatasetContext(
        tables=tables,
        relationships=[
            Relationship(e.from_table, e.from_columns, e.to_table, e.to_columns, e.source)
            for e in graph.edges
        ],
        notes=list(graph.warnings),
    )
