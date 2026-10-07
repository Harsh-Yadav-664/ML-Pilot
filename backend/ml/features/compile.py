"""Compile a ``FeatureIR`` to point-in-time-safe SQL (#100).

The output is a feature query in the contract of ``ml.tasks.pit_guard``: it reads the relation
``__labels(entity_id, cutoff_time)`` and returns ``entity_id, cutoff_time, value``, one row per
label row. The shape is always the same:

    SELECT l.entity_id, l.cutoff_time, <aggregate> AS value
    FROM __labels l
    LEFT JOIN <table> t1 ON <join> AND t1.<time> < l.cutoff_time [AND <window>]
    LEFT JOIN ... GROUP BY l.entity_id, l.cutoff_time

Every table on the path that has an event time is bounded by the cutoff, the window is a lower
bound on the event time, and the filter sits in the join so entities with no matching rows keep
their row (count 0, other aggregates NULL). Identifiers come from the schema graph and literals
are typed and quoted by sqlglot; nothing from the IR is pasted into the SQL as text.

``compile`` runs the point-in-time guard on its own output as a second, independent check.
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

from ml.data.schema_graph import SchemaGraph, Table, type_kind
from ml.features.ir import FeatureIR, FeatureIRError, FieldError, Predicate, Resolved, resolve
from ml.tasks.pit_guard import CUTOFF, ENTITY, LABELS, check

DIALECTS = ("duckdb", "postgres")


class _Sql:
    def __init__(self, dialect: str) -> None:
        if dialect not in DIALECTS:
            raise ValueError(f"dialect must be one of {DIALECTS}, not {dialect!r}")
        self.dialect = dialect
        self.keywords = Dialect.get_or_raise(dialect).tokenizer_class.KEYWORDS

    def ident(self, name: str) -> str:
        bare = name.isidentifier() and name == name.lower() and name.isascii()
        if bare and name.upper() not in self.keywords:
            return name
        return exp.to_identifier(name, quoted=True).sql(self.dialect)

    def table(self, table: Table) -> str:
        if "." in table.key:
            return f"{self.ident(table.db_schema)}.{self.ident(table.name)}"
        return self.ident(table.name)

    def literal(self, value: object, kind: str) -> str:
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, int | float) and kind in ("integer", "float"):
            return repr(value)
        if kind in ("date", "timestamp"):
            return str(
                exp.cast(
                    exp.Literal.string(str(value)), "date" if kind == "date" else "timestamp"
                ).sql(self.dialect)
            )
        return str(exp.Literal.string(str(value)).sql(self.dialect))


def compile(
    ir: FeatureIR, graph: SchemaGraph, dialect: str = "duckdb", *, check_guard: bool = True
) -> str:
    """SQL for ``ir``. Raises ``FeatureIRError`` with field-level errors if it does not fit."""
    sql = _Sql(dialect)
    resolved = resolve(ir, graph)
    if ir.ratio_to is None:
        query = _plain(ir, resolved, sql)
    else:
        top = ir.model_copy(update={"ratio_to": None, "transform": "none"})
        bottom = resolve(ir.ratio_to, graph)
        query = _ratio(ir, _plain(top, resolved, sql), _plain(ir.ratio_to, bottom, sql))
    if check_guard:
        result = check(query, graph, dialect, allow_rewrite=False)
        if result.status != "accepted":
            raise FeatureIRError(
                [FieldError("sql", f"{r.code}: {r.message}") for r in result.reasons]
                or [FieldError("sql", "the point-in-time guard did not accept the compiled SQL")]
            )
    return query


def _plain(ir: FeatureIR, resolved: Resolved, sql: _Sql) -> str:
    value = _value(ir, resolved, sql)
    return _select(ir, resolved, sql, value)


def _select(ir: FeatureIR, resolved: Resolved, sql: _Sql, value: str) -> str:
    lines = [
        f"SELECT l.{ENTITY}, l.{CUTOFF}, {value} AS value",
        f"FROM {LABELS} l",
    ]
    for i, hop in enumerate(resolved.hops):
        alias = f"t{i + 1}"
        here = "l." + ENTITY if i == 0 else f"t{i}.{sql.ident(hop.here_column)}"
        table = hop.table
        on = [f"{alias}.{sql.ident(hop.there_column)} = {here}"]
        if table.time_column:
            time = f"{alias}.{sql.ident(table.time_column)}"
            on.append(f"{time} < l.{CUTOFF}")
            if resolved.event is hop and ir.window_days:
                on.append(f"{time} >= l.{CUTOFF} - INTERVAL '{int(ir.window_days)} days'")
        if i == len(resolved.hops) - 1 and ir.agg != "share":
            on.extend(_condition(p, alias, table, sql) for p in ir.filter)
        lines.append(f"LEFT JOIN {sql.table(table)} {alias}")
        lines.extend(f"  {'ON' if j == 0 else 'AND'} {term}" for j, term in enumerate(on))
    lines.append(f"GROUP BY l.{ENTITY}, l.{CUTOFF}")
    return "\n".join(lines)


def _value(ir: FeatureIR, resolved: Resolved, sql: _Sql) -> str:
    n = len(resolved.hops)
    source = resolved.hops[-1]
    alias = f"t{n}"
    if ir.agg == "count":
        expr = f"count({alias}.{sql.ident(source.there_column)})"
    elif ir.agg == "count_distinct":
        assert ir.column is not None
        expr = f"count(DISTINCT {alias}.{sql.ident(ir.column)})"
    elif ir.agg == "sum":
        assert ir.column is not None
        expr = f"coalesce(sum({alias}.{sql.ident(ir.column)}), 0)"
    elif ir.agg in ("mean", "min", "max", "std"):
        assert ir.column is not None
        fn = {"mean": "avg", "min": "min", "max": "max", "std": "stddev_samp"}[ir.agg]
        expr = f"{fn}({alias}.{sql.ident(ir.column)})"
    elif ir.agg == "share":
        cond = " AND ".join(_condition(p, alias, source.table, sql) for p in ir.filter)
        expr = f"avg(CASE WHEN {cond} THEN 1.0 ELSE 0.0 END)"
    else:
        assert resolved.event is not None
        index = resolved.hops.index(resolved.event) + 1
        time = f"t{index}.{sql.ident(resolved.event.table.time_column or '')}"
        edge = f"max({time})" if ir.agg == "days_since_last" else f"min({time})"
        if sql.dialect == "postgres":
            expr = f"EXTRACT(EPOCH FROM (CAST(l.{CUTOFF} AS TIMESTAMP) - CAST({edge} AS TIMESTAMP))) / 86400.0"
        else:
            expr = f"date_diff('second', CAST({edge} AS TIMESTAMP), CAST(l.{CUTOFF} AS TIMESTAMP)) / 86400.0"
    if ir.transform == "log1p":
        expr = f"ln(1 + {expr})"
    return expr


def _condition(pred: Predicate, alias: str, table: Table, sql: _Sql) -> str:
    column = next(c for c in table.columns if c.name == pred.column)
    kind = type_kind(column.type)
    ref = f"{alias}.{sql.ident(pred.column)}"
    if pred.op == "is_null":
        return f"{ref} IS NULL"
    if pred.op == "is_not_null":
        return f"{ref} IS NOT NULL"
    if pred.op == "in":
        assert isinstance(pred.value, list)
        return f"{ref} IN ({', '.join(sql.literal(v, kind) for v in pred.value)})"
    op = "<>" if pred.op == "!=" else pred.op
    return f"{ref} {op} {sql.literal(pred.value, kind)}"


def _ratio(ir: FeatureIR, numerator: str, denominator: str) -> str:
    def indent(text: str) -> str:
        return "\n".join("  " + line for line in text.splitlines())

    expr = "1.0 * a.value / NULLIF(b.value, 0)"
    if ir.transform == "log1p":
        expr = f"ln(1 + {expr})"
    join = f"ON {{t}}.{ENTITY} = l.{ENTITY} AND {{t}}.{CUTOFF} = l.{CUTOFF}"
    return "\n".join(
        [
            "WITH a AS (",
            indent(numerator),
            "), b AS (",
            indent(denominator),
            ")",
            f"SELECT l.{ENTITY}, l.{CUTOFF}, {expr} AS value",
            f"FROM {LABELS} l",
            "LEFT JOIN a " + join.format(t="a"),
            "LEFT JOIN b " + join.format(t="b"),
        ]
    )


def compile_entity_value(
    graph: SchemaGraph,
    entity_table: str,
    column: str | None,
    dialect: str = "duckdb",
    *,
    created_at: str | None = None,
    check_guard: bool = True,
) -> str:
    """SQL for a feature of the entity row itself: its ``column``, or, with ``created_at``, its
    age in days at the cutoff. The entity table is bounded by the cutoff like any other table.
    """
    sql = _Sql(dialect)
    table = next((t for t in graph.tables if t.key == entity_table), None)
    errors: list[FieldError] = []
    if table is None:
        errors.append(FieldError("entity_table", f"{entity_table!r} is not a table of the schema"))
    elif len(table.primary_key) != 1:
        errors.append(FieldError("entity_table", "the entity table needs a single-column key"))
    elif not table.time_column and table.is_static is not True:
        errors.append(
            FieldError(
                "entity_table",
                f"table {entity_table!r} has no event time and is not confirmed as static",
            )
        )
    elif (column is None) == (created_at is None):
        errors.append(FieldError("column", "give either a column or created_at"))
    else:
        names = {c.name for c in table.columns}
        for field, name in (("column", column), ("created_at", created_at)):
            if name is not None and name not in names:
                errors.append(FieldError(field, f"{entity_table!r} has no column {name!r}"))
    if errors or table is None:
        raise FeatureIRError(errors)
    if created_at is not None:
        time = f"e.{sql.ident(created_at)}"
        if sql.dialect == "postgres":
            value = (
                f"EXTRACT(EPOCH FROM (CAST(l.{CUTOFF} AS TIMESTAMP) - CAST({time} AS TIMESTAMP)))"
                " / 86400.0"
            )
        else:
            value = (
                f"date_diff('second', CAST({time} AS TIMESTAMP), "
                f"CAST(l.{CUTOFF} AS TIMESTAMP)) / 86400.0"
            )
    else:
        assert column is not None
        value = f"e.{sql.ident(column)}"
    on = [f"e.{sql.ident(table.primary_key[0])} = l.{ENTITY}"]
    if table.time_column:
        on.append(f"e.{sql.ident(table.time_column)} < l.{CUTOFF}")
    lines = [
        f"SELECT l.{ENTITY}, l.{CUTOFF}, {value} AS value",
        f"FROM {LABELS} l",
        f"LEFT JOIN {sql.table(table)} e",
        *(f"  {'ON' if j == 0 else 'AND'} {term}" for j, term in enumerate(on)),
    ]
    query = "\n".join(lines)
    if check_guard:
        result = check(query, graph, dialect, allow_rewrite=False)
        if result.status != "accepted":
            raise FeatureIRError(
                [FieldError("sql", f"{r.code}: {r.message}") for r in result.reasons]
                or [FieldError("sql", "the point-in-time guard did not accept the compiled SQL")]
            )
    return query
