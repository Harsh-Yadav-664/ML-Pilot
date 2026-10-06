"""The schema graph of a live database: tables, columns, keys, event times (issue #45).

The task builder and the feature agent need to know how tables relate and which column says
*when* something happened. This module reads the catalog of a Postgres, SQLite or DuckDB
source and returns a JSON-serialisable ``SchemaGraph``:

* every table with its row count, primary key, columns and a semantic hint per column;
* the time column of each table (typed timestamps first, then column-name tokens), with a
  leakage hint when the only candidate is a last-modified time such as ``updated_at``;
* the edges between tables: declared foreign keys, plus keys inferred when the database
  does not declare them (column name against table name, compatible types, and the share
  of child keys found in the parent);
* the user's overrides, applied on top and echoed back.

Everything MLPilot asks the database goes through ``sql_guard.catalog``, so the introspection
runs in the same read-only session, with the same limits, as any other query. Nothing here
reads cell values into a prompt: the overlap check returns two counts per candidate.

Per-column statistics live in ``ml.data.profiling.db_stats`` (#96); ``refine_hints`` copies the
semantic types they find (``categorical`` needs cardinalities) onto the graph's columns.
"""

from __future__ import annotations

import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from ml.data import sql_guard
from ml.data.engine import EngineError, TableRef, TableSchema, quote_ident
from ml.data.sources.base import ConnectionFailure, SourceWithChecks

DEFAULT_SCHEMAS = ("main", "public")
OVERLAP_THRESHOLD = 0.95  # share of child keys that must exist in the parent
OVERLAP_SAMPLE_ROWS = 20_000  # child rows read per candidate (the first rows, not a random draw)
MAX_OVERLAP_CHECKS = 400  # candidates checked per graph; the rest are reported, not guessed
ESTIMATE_ABOVE_ROWS = 1_000_000  # Postgres only: below this the exact count is cheap

Hint = Literal["id", "time", "numeric", "categorical", "boolean", "text", "other"]
Kind = Literal["integer", "float", "text", "boolean", "timestamp", "date", "other"]
EdgeSource = Literal["declared", "inferred", "user"]
Cardinality = Literal["1:1", "1:N"]


class Column(BaseModel):
    name: str
    type: str
    nullable: bool
    hint: Hint = Field(
        description="id, time, numeric, boolean, text or other from the declared type and name; "
        "categorical (and a sharper id/text/boolean) once column statistics exist (#96)"
    )
    is_primary_key: bool = False


class Table(BaseModel):
    key: str = Field(description="Name used in edges and overrides: 'orders', or 'schema.orders'")
    name: str
    db_schema: str
    row_count: int
    row_count_estimated: bool = False
    primary_key: list[str]
    columns: list[Column]
    time_column: str | None = Field(None, description="The column that says when a row happened")
    time_column_source: Literal["inferred", "user"] | None = None
    time_candidates: list[str] = Field(default_factory=list, description="Best first")
    time_leakage_hint: str | None = Field(
        None,
        description="Set when the chosen time column is a last-modified time: rows are "
        "rewritten after the event, so it can leak the future (see #54)",
    )
    is_static: bool | None = Field(
        None, description="Confirmed by the user as a table without event times; None = unknown"
    )


class Edge(BaseModel):
    from_table: str
    from_columns: list[str]
    to_table: str
    to_columns: list[str]
    source: EdgeSource
    confidence: float = Field(description="1.0 for declared and user edges")
    cardinality: Cardinality = Field(
        description="1:1 when the child columns are the child's primary key, else 1:N"
    )
    overlap: float | None = Field(
        None, description="Inferred edges: share of sampled child keys found in the parent"
    )


class EdgeRef(BaseModel):
    """Names an edge in overrides."""

    from_table: str
    from_columns: list[str] = Field(min_length=1)
    to_table: str
    to_columns: list[str] = Field(min_length=1)

    def matches(self, edge: Edge) -> bool:
        return (
            edge.from_table == self.from_table
            and edge.to_table == self.to_table
            and edge.from_columns == self.from_columns
            and edge.to_columns == self.to_columns
        )


class SchemaOverrides(BaseModel):
    """What the user decided. In a PATCH, a field that is given replaces the stored one."""

    time_columns: dict[str, str | None] | None = Field(
        None, description="table -> column; null says the table has no time column"
    )
    static_tables: dict[str, bool] | None = Field(None, description="table -> is_static")
    add_edges: list[EdgeRef] | None = None
    remove_edges: list[EdgeRef] | None = None


class SchemaGraph(BaseModel):
    dialect: str
    tables: list[Table]
    edges: list[Edge]
    overrides: SchemaOverrides = Field(default_factory=SchemaOverrides)
    warnings: list[str] = Field(
        default_factory=list,
        description="Things that could not be checked or no longer apply; never silent",
    )


class OverrideError(ValueError):
    """An override names a table or column that does not exist."""


# -- column kinds and hints --------------------------------------------------------------


def type_kind(type_name: str) -> Kind:
    t = type_name.lower()
    if "timestamp" in t or "datetime" in t:
        return "timestamp"
    if t.startswith("date"):
        return "date"
    if "bool" in t:
        return "boolean"
    if "int" in t or "serial" in t:
        return "integer"
    if any(w in t for w in ("float", "double", "real", "numeric", "decimal")):
        return "float"
    if any(w in t for w in ("char", "text", "string", "uuid", "clob")):
        return "text"
    return "other"


ID_NAME = re.compile(
    r"^(?:id|(?P<stem>.+?)(?:_id|_key|_fk|_uuid|(?<=[a-z0-9])ID|(?<=[a-z0-9])Id))$"
)
LOOSE_ID_NAME = re.compile(r"^(?P<stem>[a-z]{3,})id$")  # customerid: only trusted for inference


def id_stem(column: str, *, loose: bool = False) -> str | None:
    """``customer_id`` -> ``customer``; None if the name does not look like a key.

    ``loose`` also accepts an all-lowercase ``customerid``. That form is only safe where a
    table of that name must exist as well (inference), not as a hint on its own (``paid``).
    """
    match = ID_NAME.match(column)
    if match is None and loose:
        match = LOOSE_ID_NAME.match(column)
    if match is None:
        return None
    stem = (match.group("stem") or "").strip("_").lower()
    return stem or None


def _looks_like_id(column: str) -> bool:
    return column.lower() == "id" or id_stem(column) is not None


def column_hint(column: str, kind: Kind, *, is_key: bool) -> Hint:
    if kind in ("timestamp", "date"):
        return "time"
    if kind == "boolean":
        return "boolean"
    if kind in ("integer", "text") and (is_key or _looks_like_id(column)):
        return "id"
    if kind in ("integer", "float"):
        return "numeric"
    if kind == "text":
        return "text"
    return "other"


# -- time columns ------------------------------------------------------------------------

TIME_SUFFIX = re.compile(r"(?:^|_)(?:at|on|date|time|ts|timestamp|datetime|epoch|utc)$")
UPDATE_TOKENS = ("updated", "modified", "changed", "last_seen", "synced", "refreshed")
EVENT_TOKENS = (
    "created",
    "ordered",
    "placed",
    "purchased",
    "occurred",
    "event",
    "started",
    "opened",
    "sent",
    "signup",
    "registered",
    "logged",
    "timestamp",
    "refunded",
    "paid",
)
END_TOKENS = ("resolved", "closed", "ended", "completed", "cancelled", "canceled", "deleted")
ATTRIBUTE_TOKENS = ("birth", "dob", "expir", "valid", "until", "due")


def _is_update_name(name: str) -> bool:
    return any(token in name for token in UPDATE_TOKENS)


def time_candidates(columns: list[Column]) -> list[tuple[int, str]]:
    """(score, column) for every column that may be the row's time, best first.

    Typed timestamps and dates score highest; other columns qualify by name (``created_at``
    stored as text or epoch seconds). Creation and event names beat end times, which beat
    attributes (``birth_date``), which beat last-modified times.
    """
    scored: list[tuple[int, int, str]] = []
    for position, col in enumerate(columns):
        name = col.name.lower()
        kind = type_kind(col.type)
        typed = kind in ("timestamp", "date")
        named = (bool(TIME_SUFFIX.search(name)) or name in EVENT_TOKENS) and kind in (
            "text",
            "integer",
            "float",
        )
        if not (typed or named):
            continue
        score = 100 if typed else 50
        if any(t in name for t in EVENT_TOKENS):
            score += 20
        if any(t in name for t in END_TOKENS):
            score -= 15
        if any(t in name for t in ATTRIBUTE_TOKENS):
            score -= 30
        if _is_update_name(name):
            score -= 60
        scored.append((score, position, col.name))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(score, name) for score, _, name in scored]


def update_time_hint(column: str) -> str | None:
    if not _is_update_name(column.lower()):
        return None
    return (
        f"'{column}' is a last-modified time: a row is rewritten after the event it describes, "
        "so using it as the event time can leak the future into training."
    )


# -- names -------------------------------------------------------------------------------


def singular(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return w


def name_strength(stem: str, table_name: str) -> float:
    """How well a key's stem names a table: 1.0 exact, 0.7 as a role or prefix, else 0."""
    s, t = singular(stem), singular(table_name)
    if s == t:
        return 1.0
    if t.endswith("_" + s) or s.endswith("_" + t):
        return 0.7
    return 0.0


def table_key(table: TableRef) -> str:
    return table.name if table.schema in DEFAULT_SCHEMAS else f"{table.schema}.{table.name}"


# -- reading the catalog -----------------------------------------------------------------


def build_schema_graph(
    source: SourceWithChecks,
    overrides: SchemaOverrides | None = None,
    *,
    max_overlap_checks: int = MAX_OVERLAP_CHECKS,
) -> SchemaGraph:
    """Introspect ``source`` and return its graph with ``overrides`` applied.

    Raises ``OverrideError`` if an override names something that is not in the database
    (checked by ``validate_overrides``, which the API calls before it saves anything).
    """
    warnings: list[str] = []
    refs = source.list_tables()
    schemas = {table_key(ref): source.table_schema(ref) for ref in refs}
    tables = [_table(source, schema, warnings) for schema in schemas.values()]
    by_key = {t.key: t for t in tables}

    edges = _declared_edges(schemas, by_key, warnings)
    edges += _inferred_edges(source, by_key, edges, warnings, max_overlap_checks)
    graph = SchemaGraph(dialect=source.dialect, tables=tables, edges=edges, warnings=warnings)
    return apply_overrides(graph, overrides or SchemaOverrides())


def _table(source: SourceWithChecks, schema: TableSchema, warnings: list[str]) -> Table:
    ref = schema.table
    key = table_key(ref)
    pk = set(schema.primary_key)
    fk_columns = {c for fk in schema.foreign_keys for c in fk.columns}
    columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            hint=column_hint(
                c.name, type_kind(c.type), is_key=c.name in pk or c.name in fk_columns
            ),
            is_primary_key=c.name in pk,
        )
        for c in schema.columns
    ]
    count, estimated = _row_count(source, ref, warnings)
    candidates = time_candidates(columns)
    time_column = candidates[0][1] if candidates else None
    return Table(
        key=key,
        name=ref.name,
        db_schema=ref.schema,
        row_count=count,
        row_count_estimated=estimated,
        primary_key=list(schema.primary_key),
        columns=columns,
        time_column=time_column,
        time_column_source="inferred" if time_column else None,
        time_candidates=[name for _, name in candidates],
        time_leakage_hint=update_time_hint(time_column) if time_column else None,
    )


def table_row_count(source: SourceWithChecks, ref: TableRef) -> tuple[int, bool]:
    """(rows, estimated). Postgres tables over a million rows use the planner's estimate; every
    other count is exact. Raises ``EngineError`` / ``ConnectionFailure`` if the count fails."""
    if source.dialect == "postgres":
        rows = sql_guard.catalog(
            source,
            "SELECT c.reltuples::bigint FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :s AND c.relname = :t",
            {"s": ref.schema, "t": ref.name},
        )
        estimate = int(rows[0][0]) if rows and rows[0][0] is not None else -1
        if estimate >= ESTIMATE_ABOVE_ROWS:
            return estimate, True
    rows = sql_guard.catalog(source, f"SELECT count(*) FROM {_qualified(ref)}")
    return int(rows[0][0]), False


def _row_count(source: SourceWithChecks, ref: TableRef, warnings: list[str]) -> tuple[int, bool]:
    try:
        return table_row_count(source, ref)
    except (EngineError, ConnectionFailure) as e:
        warnings.append(f"Could not count the rows of {table_key(ref)}: {e}")
        return 0, False


def _qualified(ref: TableRef) -> str:
    return f"{quote_ident(ref.schema)}.{quote_ident(ref.name)}"


def _declared_edges(
    schemas: dict[str, TableSchema], by_key: dict[str, Table], warnings: list[str]
) -> list[Edge]:
    edges: list[Edge] = []
    for key, schema in schemas.items():
        for fk in schema.foreign_keys:
            target = table_key(fk.ref_table)
            if target not in by_key:
                warnings.append(
                    f"{key}.{','.join(fk.columns)} references {target}, which is not listed"
                )
                continue
            edges.append(
                Edge(
                    from_table=key,
                    from_columns=list(fk.columns),
                    to_table=target,
                    to_columns=list(fk.ref_columns),
                    source="declared",
                    confidence=1.0,
                    cardinality=_cardinality(by_key[key], fk.columns),
                )
            )
    return edges


def _cardinality(child: Table, columns: list[str]) -> Cardinality:
    return "1:1" if child.primary_key and set(columns) == set(child.primary_key) else "1:N"


def _inferred_edges(
    source: SourceWithChecks,
    by_key: dict[str, Table],
    declared: list[Edge],
    warnings: list[str],
    max_checks: int,
) -> list[Edge]:
    covered = {(e.from_table, tuple(e.from_columns)) for e in declared}
    checks = 0
    inferred: list[Edge] = []
    for child in by_key.values():
        for col in child.columns:
            stem = id_stem(col.name, loose=True)
            kind = type_kind(col.type)
            if (
                stem is None
                or kind not in ("integer", "text")
                or (child.key, (col.name,)) in covered
            ):
                continue
            candidates = _parent_candidates(by_key, child, col, stem, kind)
            best: tuple[float, float, Edge] | None = None
            for parent, parent_col, strength in candidates:
                if checks >= max_checks:
                    break
                checks += 1
                try:
                    overlap = _overlap(source, child, col.name, parent, parent_col)
                    unique = parent_col in parent.primary_key and len(parent.primary_key) == 1
                    unique = unique or _is_unique(source, parent, parent_col)
                except (EngineError, ConnectionFailure) as e:
                    warnings.append(
                        f"Could not check {child.key}.{col.name} against {parent.key}.{parent_col}: {e}"
                    )
                    continue
                if overlap is None or overlap < OVERLAP_THRESHOLD or not unique:
                    continue
                edge = Edge(
                    from_table=child.key,
                    from_columns=[col.name],
                    to_table=parent.key,
                    to_columns=[parent_col],
                    source="inferred",
                    confidence=round(0.5 * strength + 0.5 * overlap, 3),
                    cardinality=_cardinality(child, [col.name]),
                    overlap=round(overlap, 4),
                )
                if best is None or (strength, overlap) > best[:2]:
                    best = (strength, overlap, edge)
            if best is not None:
                inferred.append(best[2])
    if checks >= max_checks:
        warnings.append(
            f"Inference stopped after {max_checks} candidate checks; "
            "add the remaining relationships by hand."
        )
    return inferred


def _parent_candidates(
    by_key: dict[str, Table], child: Table, col: Column, stem: str, kind: Kind
) -> list[tuple[Table, str, float]]:
    found: list[tuple[Table, str, float]] = []
    for parent in by_key.values():
        if parent.key == child.key:
            continue
        strength = name_strength(stem, parent.name)
        if strength == 0.0:
            continue
        names = {c.name.lower(): c for c in parent.columns}
        parent_col: Column | None = None
        if len(parent.primary_key) == 1:
            parent_col = names.get(parent.primary_key[0].lower())
        if parent_col is None:
            for guess in (col.name.lower(), "id", f"{singular(parent.name)}_id", f"{stem}_id"):
                if guess in names:
                    parent_col = names[guess]
                    break
        if parent_col is not None and type_kind(parent_col.type) == kind:
            found.append((parent, parent_col.name, strength))
    return found


def _overlap(
    source: SourceWithChecks, child: Table, column: str, parent: Table, parent_col: str
) -> float | None:
    """Share of the child's distinct non-null keys (first rows only) that exist in the parent."""
    child_ref = TableRef(child.name, child.db_schema)
    parent_ref = TableRef(parent.name, parent.db_schema)
    sql = (
        "SELECT count(*) AS total, "
        "sum(CASE WHEN EXISTS (SELECT 1 FROM "
        f"{_qualified(parent_ref)} AS p WHERE p.{quote_ident(parent_col)} = ch.k) "
        "THEN 1 ELSE 0 END) AS matched "
        f"FROM (SELECT DISTINCT k FROM (SELECT {quote_ident(column)} AS k "
        f"FROM {_qualified(child_ref)} WHERE {quote_ident(column)} IS NOT NULL "
        f"LIMIT {OVERLAP_SAMPLE_ROWS}) AS s) AS ch"
    )
    ((total, matched),) = sql_guard.catalog(source, sql)
    if not total:
        return None
    return float(matched or 0) / float(total)


def _is_unique(source: SourceWithChecks, parent: Table, column: str) -> bool:
    ref = TableRef(parent.name, parent.db_schema)
    col = quote_ident(column)
    ((non_null, distinct),) = sql_guard.catalog(
        source, f"SELECT count({col}), count(DISTINCT {col}) FROM {_qualified(ref)}"
    )
    return bool(non_null) and non_null == distinct


# -- overrides ---------------------------------------------------------------------------


def validate_overrides(graph: SchemaGraph, overrides: SchemaOverrides) -> None:
    """Raise ``OverrideError`` if an override names a table or column the graph does not have."""
    tables = {t.key: t for t in graph.tables}

    def need_columns(table: str, columns: list[str]) -> None:
        if table not in tables:
            raise OverrideError(f"Unknown table {table!r}")
        known = {c.name for c in tables[table].columns}
        missing = [c for c in columns if c not in known]
        if missing:
            raise OverrideError(f"Table {table!r} has no column {missing[0]!r}")

    for table, column in (overrides.time_columns or {}).items():
        need_columns(table, [column] if column else [])
    for table in overrides.static_tables or {}:
        need_columns(table, [])
    for ref in (overrides.add_edges or []) + (overrides.remove_edges or []):
        need_columns(ref.from_table, ref.from_columns)
        need_columns(ref.to_table, ref.to_columns)
        if len(ref.from_columns) != len(ref.to_columns):
            raise OverrideError("An edge needs as many columns on both sides")


def apply_overrides(graph: SchemaGraph, overrides: SchemaOverrides) -> SchemaGraph:
    """The graph with overrides applied. Overrides that no longer match the database (it
    changed since they were saved) are skipped and reported in ``warnings``."""
    warnings = list(graph.warnings)
    tables = {t.key: t.model_copy(deep=True) for t in graph.tables}

    def column_names(table: str) -> set[str]:
        return {c.name for c in tables[table].columns}

    for table, column in (overrides.time_columns or {}).items():
        if table not in tables or (column and column not in column_names(table)):
            warnings.append(f"Ignored the saved time column of {table!r}: it is no longer there")
            continue
        t = tables[table]
        t.time_column, t.time_column_source = column, "user"
        t.time_leakage_hint = update_time_hint(column) if column else None
    for table, static in (overrides.static_tables or {}).items():
        if table in tables:
            tables[table].is_static = static
        else:
            warnings.append(f"Ignored the saved static flag of {table!r}: it is no longer there")

    edges = list(graph.edges)
    for ref in overrides.remove_edges or []:
        edges = [e for e in edges if not ref.matches(e)]
    for ref in overrides.add_edges or []:
        if any(ref.matches(e) for e in edges):
            continue
        if not _edge_exists(tables, ref):
            warnings.append(
                f"Ignored the saved edge {ref.from_table} -> {ref.to_table}: it is no longer valid"
            )
            continue
        edges.append(
            Edge(
                from_table=ref.from_table,
                from_columns=ref.from_columns,
                to_table=ref.to_table,
                to_columns=ref.to_columns,
                source="user",
                confidence=1.0,
                cardinality=_cardinality(tables[ref.from_table], ref.from_columns),
            )
        )
    return SchemaGraph(
        dialect=graph.dialect,
        tables=[tables[t.key] for t in graph.tables],
        edges=edges,
        overrides=overrides,
        warnings=warnings,
    )


def _edge_exists(tables: dict[str, Table], ref: EdgeRef) -> bool:
    for table, columns in ((ref.from_table, ref.from_columns), (ref.to_table, ref.to_columns)):
        if table not in tables or not set(columns) <= {c.name for c in tables[table].columns}:
            return False
    return True


def refine_hints(graph: SchemaGraph, semantic_types: dict[str, dict[str, str]]) -> SchemaGraph:
    """The graph with column hints replaced by statistics-based semantic types.

    ``semantic_types`` maps a table key to {column: semantic type}; tables or columns it does
    not mention keep their declared-type hints."""
    tables = [t.model_copy(deep=True) for t in graph.tables]
    for table in tables:
        for column in table.columns:
            found = semantic_types.get(table.key, {}).get(column.name)
            if found in get_args(Hint):
                column.hint = found  # type: ignore[assignment]
    return graph.model_copy(update={"tables": tables})


def merge_overrides(stored: dict[str, Any] | None, patch: SchemaOverrides) -> SchemaOverrides:
    """The stored overrides with every field given in ``patch`` replaced."""
    current = SchemaOverrides.model_validate(stored or {})
    given = {name: getattr(patch, name) for name in SchemaOverrides.model_fields}
    return current.model_copy(update={k: v for k, v in given.items() if v is not None})
