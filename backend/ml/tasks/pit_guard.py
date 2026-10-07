"""Point-in-time guard (#51): a feature query may only read rows from before the row's cutoff.

A feature query gets a relation ``__labels(entity_id, cutoff_time)`` (one row per label row,
see ``labels.py``) and returns ``entity_id, cutoff_time, value``. ``check`` decides whether the
query can leak the future, by reading its SQL, not by trusting its author:

``accepted``  every table with an event time is bounded by the cutoff.
``rewritten`` it was not, but the bound could be added safely; the changed SQL is returned.
``rejected``  it could read rows from after the cutoff, or the guard cannot tell. Reasons name
              the scope and the table.

The rules (``docs/pit_guard.md`` has the reasoning and the known limits):

1. Parse with sqlglot and qualify every column, so each reference names its table.
2. Every table source with an event time needs, in the same scope, a top-level ``AND`` term of
   its ``WHERE`` or its own ``JOIN ... ON`` of the form ``t.time < K`` or
   ``t.time <= K - INTERVAL 'n ...'`` (``n`` > 0), where ``K`` is the cutoff of the row being
   computed. A table without an event time is allowed only if the user confirmed it as static.
3. *Which* cutoff counts: ``K`` must be ``__labels.cutoff_time`` or a pass-through of it. Each
   scope's cutoff sources (its own ``__labels``, derived tables and CTEs that expose the cutoff,
   and the enclosing row for correlated subqueries) must be joined by ``cutoff_time`` equality,
   so a derived table can never supply values computed for a different cutoff. Aggregates,
   ``DISTINCT``, windows and ``LIMIT`` in such a scope must keep the cutoff in their keys, so
   they never mix label rows.
4. Refused outright: unknown tables and label tables, non-deterministic functions, frames with
   ``FOLLOWING``, set operations, lateral joins, ``USING`` joins, right and full joins.

Anything the guard does not model is rejected, not accepted. It is one layer: the runtime check
in ``pit_verify.py`` recomputes features on data truncated at each cutoff and compares.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

from ml.data import sql_guard
from ml.data.schema_graph import SchemaGraph, Table, type_kind

LABELS = "__labels"
ENTITY = "entity_id"
CUTOFF = "cutoff_time"
VALUE = "value"
OUTPUT_COLUMNS = (ENTITY, CUTOFF, VALUE)

Status = Literal["accepted", "rewritten", "rejected"]

_NONDETERMINISTIC = {
    "now",
    "current_date",
    "current_time",
    "current_timestamp",
    "current_datetime",
    "localtime",
    "localtimestamp",
    "today",
    "sysdate",
    "getdate",
    "clock_timestamp",
    "statement_timestamp",
    "transaction_timestamp",
    "timeofday",
    "random",
    "rand",
    "uuid",
    "gen_random_uuid",
    "uuid_generate_v4",
    "setseed",
    "nextval",
    "currval",
    "txid_current",
    "get_current_timestamp",
    "get_current_time",
    "uuidv4",
    "uuidv7",
    "random_uuid",
    "current_setting",
    "current_query",
}
_ANONYMOUS_OK = {"date_part"}
_MOVING_LITERALS = {"now", "today", "tomorrow", "yesterday", "epoch", "infinity", "-infinity"}
_TIMESTAMP_TYPES = {
    exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ,
    exp.DataType.Type.TIMESTAMPLTZ,
    exp.DataType.Type.TIMESTAMPNTZ,
    exp.DataType.Type.DATETIME,
}
_INTERVAL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")

Terminal = tuple[int, str, str]  # (id of the owning SELECT, source alias, column)


@dataclass(frozen=True)
class Reason:
    code: str
    message: str
    scope: str = "query"
    table: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "scope": self.scope,
            "table": self.table,
        }


@dataclass(frozen=True)
class PitResult:
    status: Status
    reasons: list[Reason]
    sql: str | None  # the checked (and possibly rewritten) query; None if rejected
    rewrites: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)  # static tables read without a bound
    warnings: list[str] = field(default_factory=list)  # for the evidence report (#54)

    @property
    def ok(self) -> bool:
        return self.status != "rejected"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reasons": [r.as_dict() for r in self.reasons],
            "sql": self.sql,
            "rewrites": self.rewrites,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
        }


class _Reject(Exception):
    """Stops the analysis at a problem that makes further analysis meaningless."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason.message)
        self.reason = reason


# -- helpers ---------------------------------------------------------------------------------


def _lc(name: str) -> str:
    return name.lower()


def _conjuncts(node: exp.Expr | None) -> list[exp.Expr]:
    if node is None:
        return []
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _unwrap(node: exp.Expr, *, date_ok: bool = False) -> exp.Expr:
    """Parentheses and casts to a plain timestamp type around a time value.

    A cast to DATE loosens a bound (``date(t) < cutoff`` keeps events later on the cutoff day) and
    a cast with a precision (``TIMESTAMP(0)``) rounds up; neither is transparent. ``date_ok``
    lets a DATE cast through, for the date-typed time column itself.
    """
    while True:
        if isinstance(node, exp.Paren):
            node = node.this
        elif isinstance(node, exp.Cast | exp.TryCast):
            to = node.args.get("to")
            if (
                isinstance(to, exp.DataType)
                and not to.expressions
                and (to.this in _TIMESTAMP_TYPES or (date_ok and to.this == exp.DataType.Type.DATE))
            ):
                node = node.this
            else:
                return node
        else:
            return node


_MICROS = {
    "microsecond": 1,
    "millisecond": 1_000,
    "second": 1_000_000,
    "minute": 60_000_000,
    "hour": 3_600_000_000,
    "day": 86_400_000_000,
    "week": 7 * 86_400_000_000,
    "month": 28 * 86_400_000_000,
    "year": 365 * 86_400_000_000,
}


def _positive_interval(node: exp.Expr) -> bool:
    """An interval of at least one microsecond (a smaller one rounds to zero in the database)."""
    if not isinstance(node, exp.Interval):
        return False
    value = node.this
    if not isinstance(value, exp.Literal):
        return False
    m = _INTERVAL.match(value.name)
    if m is None:
        return False
    unit = m.group(2)
    if not unit:
        unit_arg = node.args.get("unit")
        unit = unit_arg.name if unit_arg is not None else ""
    unit = unit.lower().rstrip("s")
    if unit not in _MICROS:
        return False
    return float(m.group(1)) * _MICROS[unit] >= 1


# -- the checker -----------------------------------------------------------------------------


class _Checker:
    def __init__(self, graph: SchemaGraph, dialect: str, allow_rewrite: bool) -> None:
        self.dialect = dialect
        self.allow_rewrite = allow_rewrite
        self.tables: dict[str, Table] = {}
        for t in graph.tables:
            self.tables.setdefault(_lc(t.name), t)
        self.reasons: list[Reason] = []
        self.rewrites: list[str] = []
        self.assumptions: list[str] = []
        self.warnings: list[str] = []
        self._derived: dict[int, _Info] = {}

    # reporting
    def reject(self, code: str, message: str, scope: str, table: str | None = None) -> None:
        reason = Reason(code, message, scope, table)
        if reason not in self.reasons:
            self.reasons.append(reason)

    # entry --------------------------------------------------------------------------------
    def run(self, sql: str) -> PitResult:
        try:
            allowed = {_lc(t.name) for t in self.tables.values()} | {LABELS}
            allowed |= {f"{_lc(t.db_schema)}.{_lc(t.name)}" for t in self.tables.values()}
            sql_guard.guard(sql, self.dialect, allowed_tables=allowed)
        except sql_guard.SqlRejected as e:
            if e.code == "table_not_allowed":  # say which kind of table it is
                try:
                    parsed = sqlglot.parse_one(sql, dialect=self.dialect)
                    if isinstance(parsed, exp.Select):
                        self._prepare(parsed)
                except _Reject as r:
                    return PitResult("rejected", [r.reason], None)
                except SqlglotError:
                    pass
            return PitResult("rejected", [Reason("sql_guard", e.reason)], None)
        try:
            tree = sqlglot.parse_one(sql, dialect=self.dialect)
        except SqlglotError as e:
            return PitResult("rejected", [Reason("parse_error", str(e).splitlines()[0])], None)
        if not isinstance(tree, exp.Select):
            return PitResult(
                "rejected",
                [
                    Reason(
                        "set_operation",
                        "a feature query is one SELECT (no UNION, INTERSECT or EXCEPT)",
                    )
                ],
                None,
            )
        try:
            self._global_checks(tree)
            tree = self._prepare(tree)
            root = build_scope(tree)
            if root is None:
                raise _Reject(Reason("parse_error", "cannot analyse the query"))
            info = self._analyze(root, "query", inherited=frozenset(), chain=[])
            self._check_output(tree, info)
        except _Reject as r:
            self.reject(r.reason.code, r.reason.message, r.reason.scope, r.reason.table)
        except SqlglotError as e:
            self.reject("parse_error", str(e).splitlines()[0], "query")
        if self.reasons:
            return PitResult(
                "rejected", self.reasons, None, self.rewrites, self.assumptions, self.warnings
            )
        final = tree.sql(dialect=self.dialect, pretty=True)
        status: Status = "rewritten" if self.rewrites else "accepted"
        return PitResult(
            status, [], final, self.rewrites, sorted(set(self.assumptions)), self.warnings
        )

    # preparation --------------------------------------------------------------------------
    def _prepare(self, tree: exp.Select) -> exp.Select:
        """Map table names to the graph, then qualify every column."""
        ctes = {_lc(c.alias) for c in tree.find_all(exp.CTE)}
        for node in list(tree.find_all(exp.Table)):
            name = _lc(node.name)
            if name in ctes and not node.args.get("db"):
                continue
            if name == LABELS:
                if node.args.get("db"):
                    raise _Reject(Reason("label_table", f"{LABELS} cannot be schema-qualified"))
                continue
            db = _lc(node.db) if node.db else None
            if db == "mlpilot" or name.startswith("labels_"):
                raise _Reject(
                    Reason(
                        "label_table",
                        f"{node.sql()} is a label table; features must not read labels",
                        table=name,
                    )
                )
            table = self.tables.get(name)
            if table is None or (
                db is not None and db not in (_lc(table.db_schema), "main", "public")
            ):
                raise _Reject(
                    Reason(
                        "unknown_table", f"{node.sql()} is not a table of the schema", table=name
                    )
                )
            node.set("db", None)
            node.set("catalog", None)
        schema: dict[str, object] = {
            name: {c.name: c.type for c in t.columns} for name, t in self.tables.items()
        }
        schema[LABELS] = {ENTITY: "bigint", CUTOFF: "timestamp"}
        return qualify(  # type: ignore[no-any-return]
            tree,
            schema=schema,
            dialect=self.dialect,
            validate_qualify_columns=True,
            quote_identifiers=False,
            identify=False,
        )

    def _global_checks(self, tree: exp.Expr) -> None:
        for node in tree.walk():
            if isinstance(node, exp.SetOperation):
                raise _Reject(
                    Reason(
                        "set_operation",
                        "set operations (UNION, INTERSECT, EXCEPT) are not supported",
                    )
                )
            if isinstance(node, exp.Lateral | exp.Unnest | exp.Pivot):
                raise _Reject(
                    Reason(
                        "lateral",
                        "lateral joins, UNNEST and PIVOT are not supported; use a correlated subquery",
                    )
                )
            if isinstance(node, exp.AnyValue | exp.First | exp.Last) or (
                isinstance(node, exp.ArrayAgg | exp.GroupConcat)
                and not isinstance(node.this, exp.Order)
            ):
                raise _Reject(
                    Reason(
                        "order_dependent",
                        f"{node.sql_name()} picks rows in an arbitrary order; use min, max or an "
                        "aggregate with ORDER BY",
                    )
                )
            if isinstance(node, exp.Func):
                name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
                if name in _NONDETERMINISTIC:
                    raise _Reject(
                        Reason(
                            "nondeterministic",
                            f"{name}() is not allowed: a feature must give the same value whenever it is computed",
                        )
                    )
            if isinstance(node, exp.Anonymous) and node.name.lower() not in _ANONYMOUS_OK:
                raise _Reject(
                    Reason(
                        "unknown_function",
                        f"{node.name}() is not a function the guard knows; it may depend on when "
                        "the query runs",
                    )
                )
            if isinstance(
                node, exp.CurrentDate | exp.CurrentTimestamp | exp.CurrentTime | exp.CurrentDatetime
            ):
                raise _Reject(
                    Reason(
                        "nondeterministic",
                        f"{node.sql()} is not allowed: a feature must give the same value whenever it is computed",
                    )
                )
            if (
                isinstance(node, exp.Literal)
                and node.is_string
                and node.name.strip().lower() in _MOVING_LITERALS
            ):
                raise _Reject(
                    Reason(
                        "nondeterministic",
                        f"the literal {node.name!r} depends on when the query runs",
                    )
                )
            if isinstance(node, exp.Window):
                spec = node.args.get("spec")
                if isinstance(spec, exp.WindowSpec) and "FOLLOWING" in (
                    str(spec.args.get("start_side") or "").upper(),
                    str(spec.args.get("end_side") or "").upper(),
                ):
                    raise _Reject(
                        Reason(
                            "following_frame",
                            "a window frame that includes FOLLOWING rows looks forward in time",
                        )
                    )
            if isinstance(node, exp.Join):
                side = str(node.args.get("side") or "").upper()
                kind = str(node.args.get("kind") or "").upper()
                method = str(node.args.get("method") or "").upper()
                if (
                    side not in ("", "LEFT", "INNER")
                    or kind not in ("", "INNER", "CROSS")
                    or method
                ):
                    shown = " ".join(x for x in (method, side, kind) if x) or "this"
                    raise _Reject(
                        Reason(
                            "outer_join",
                            f"{shown} joins are not supported; use JOIN or LEFT JOIN from __labels",
                        )
                    )
                if node.args.get("using"):
                    raise _Reject(
                        Reason(
                            "using_join",
                            "JOIN ... USING is not supported; write the condition with ON",
                        )
                    )
            if (
                isinstance(node, exp.Select)
                and node.args.get("distinct")
                and isinstance(node.args["distinct"].args.get("on"), exp.Expr)
            ):
                raise _Reject(Reason("distinct_on", "DISTINCT ON is not supported"))

    # analysis -----------------------------------------------------------------------------
    def _analyze(
        self, scope: Scope, label: str, *, inherited: frozenset[Terminal], chain: list[_Ctx]
    ) -> _Info:
        key = id(scope.expression)
        if not chain and key in self._derived:
            return self._derived[key]
        select = scope.expression
        if not isinstance(select, exp.Select):
            raise _Reject(Reason("set_operation", "set operations are not supported", label))
        ctx = _Ctx(select, key)
        chain = [*chain, ctx]
        own_labels: list[str] = []
        sources = scope.selected_sources
        from_node = select.args.get("from_") or select.args.get("from")
        in_ast = [from_node.this] if from_node is not None and from_node.this is not None else []
        in_ast += [j.this for j in select.args.get("joins") or []]
        ast_aliases = [_lc(n.alias_or_name) for n in in_ast]
        if sorted(ast_aliases) != sorted(_lc(a) for a in sources) or len(set(ast_aliases)) != len(
            ast_aliases
        ):
            raise _Reject(
                Reason(
                    "unsupported_source",
                    "a source of this scope cannot be analysed (duplicate or case-clashing "
                    "aliases, or a join kind the guard does not model)",
                    label,
                )
            )
        ctx.order = ast_aliases
        out_names = [_lc(p.alias_or_name) for p in select.expressions]
        if len(set(out_names)) != len(out_names):
            raise _Reject(
                Reason(
                    "duplicate_output_names",
                    "the SELECT list gives two columns the same name, so which one a reference "
                    "means is not safe to assume",
                    label,
                )
            )
        for alias, (_, source) in sources.items():
            a = _lc(alias)
            if isinstance(source, exp.Table):
                if _lc(source.name) == LABELS:
                    own_labels.append(a)
                    ctx.aliases[a] = "labels"
                else:
                    ctx.aliases[a] = "table"
                    ctx.tables[a] = self._graph_table(source, label)
            else:
                ctx.aliases[a] = "derived"
        if len(own_labels) > 1:
            raise _Reject(
                Reason(
                    "multiple_labels",
                    f"{LABELS} appears {len(own_labels)} times in one scope; each scope may read it once",
                    label,
                )
            )
        derived_info: dict[str, _Info] = {}
        for alias, (_, source) in sources.items():
            if isinstance(source, Scope):
                a = _lc(alias)
                info = self._analyze(source, f"{label} > {alias}", inherited=frozenset(), chain=[])
                derived_info[a] = info
                if info.has_own_cutoff and not info.cutoff_outputs:
                    raise _Reject(
                        Reason(
                            "derived_without_cutoff",
                            f"{alias!r} reads the labels but does not expose cutoff_time, so its rows cannot be tied to a cutoff",
                            label,
                        )
                    )
                ctx.derived[a] = info
        own: set[Terminal] = set()
        for a in own_labels:
            own.add((key, a, CUTOFF))
        derived_groups: list[set[Terminal]] = []
        for a, info in derived_info.items():
            group = {(key, a, col) for col in info.cutoff_outputs}
            own |= group
            if group:
                derived_groups.append(group)
        # terms: where, and each join's ON, tagged with the alias a join introduces
        terms: list[tuple[exp.Expr, str | None]] = [
            (c, None) for c in _conjuncts(select.args.get("where") and select.args["where"].this)
        ]
        join_of: dict[str, exp.Join] = {}
        for join in select.args.get("joins") or []:
            alias = _lc(join.this.alias_or_name)
            join_of[alias] = join
            for c in _conjuncts(join.args.get("on")):
                terms.append((c, alias))
        # the cutoff class of this scope
        uf = _UnionFind()
        for t in own | inherited:
            uf.add(t)
        for group in derived_groups:
            first_t, *rest = sorted(group)
            for t in rest:
                uf.union(first_t, t)
        inherited_list = sorted(inherited)
        for t in inherited_list[1:]:
            uf.union(inherited_list[0], t)
        for conj, join_alias in terms:
            if not isinstance(conj, exp.EQ):
                continue
            left = self._terminal(conj.this, chain)
            right = self._terminal(conj.expression, chain)
            if left is None or right is None or left not in uf or right not in uf:
                continue
            if join_alias is not None and join_alias not in (left[1], right[1]):
                continue  # an equality in another source's ON does not restrict this join
            uf.union(left, right)
        members = own | inherited
        if members and not uf.single_class(members):
            loose = sorted({f"{t[1]}.{t[2]}" for t in members if not uf.same(t, min(members))})
            raise _Reject(
                Reason(
                    "cutoff_mismatch",
                    "rows from different cutoffs could be combined: join these on cutoff_time: "
                    + ", ".join(loose),
                    label,
                )
            )
        info = _Info(
            has_own_cutoff=bool(own),
            cutoff_outputs={},
        )
        cl = frozenset(members)
        # tables -----------------------------------------------------------------------------
        for alias, table in ctx.tables.items():
            self._check_table(select, label, alias, table, cl, own, terms, join_of, chain)
        # keys, windows, limits ---------------------------------------------------------------
        if own:
            self._check_mixing(select, label, cl, chain)
        # outputs that pass a cutoff through ---------------------------------------------------
        for proj in select.expressions:
            inner = proj.this if isinstance(proj, exp.Alias) else proj
            term = self._terminal(inner, chain)
            if term is not None and term in cl and term in own:
                info.cutoff_outputs[_lc(proj.alias_or_name)] = term
        if (
            not own
            and (select.args.get("limit") or select.args.get("offset") or select.args.get("fetch"))
            and not select.args.get("order")
        ):
            raise _Reject(
                Reason(
                    "limit_without_order",
                    "LIMIT without ORDER BY picks arbitrary rows; order by the event time",
                    label,
                )
            )
        # nested subqueries (correlated: they see this scope's class) --------------------------
        for sub in scope.subquery_scopes:
            self._analyze(sub, f"{label} > subquery", inherited=cl, chain=chain)
        if not chain[:-1]:
            self._derived[key] = info
        return info

    def _graph_table(self, source: exp.Table, label: str) -> Table:
        table = self.tables.get(_lc(source.name))
        if table is None:
            raise _Reject(
                Reason(
                    "unknown_table",
                    f"{source.name!r} is not a table of the schema",
                    label,
                    source.name,
                )
            )
        return table

    # terminals ----------------------------------------------------------------------------
    def _terminal(self, node: exp.Expr, chain: list[_Ctx]) -> Terminal | None:
        node = _unwrap(node)
        if not isinstance(node, exp.Column) or not node.table:
            return None
        alias, col = _lc(node.table), _lc(node.name)
        for ctx in reversed(chain):
            kind = ctx.aliases.get(alias)
            if kind is None:
                continue
            if kind == "labels" and col == CUTOFF:
                return (ctx.key, alias, col)
            if kind == "derived" and col in ctx.derived[alias].cutoff_outputs:
                return (ctx.key, alias, col)
            return None  # a column of this source that is not a cutoff (shadows outer names)
        return None

    # tables -------------------------------------------------------------------------------
    def _check_table(
        self,
        select: exp.Select,
        label: str,
        alias: str,
        table: Table,
        cl: frozenset[Terminal],
        own: set[Terminal],
        terms: list[tuple[exp.Expr, str | None]],
        join_of: dict[str, exp.Join],
        chain: list[_Ctx],
    ) -> None:
        if table.is_static:
            self.assumptions.append(
                f"{table.name} is static (confirmed in the schema): read without a time bound"
            )
            if table.time_leakage_hint and table.time_column:
                warning = (
                    f"static assumption on a table that changes: {table.name} is marked static, "
                    f"but its {table.time_column} is a last-modified time, so its rows are "
                    "rewritten after the events they describe and a feature on it can use "
                    "information from after the cutoff"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)
            return
        tcol = table.time_column
        if tcol is None:
            self.reject(
                "no_time_column",
                f"{table.name} has no event-time column, so its rows cannot be limited to before the cutoff; "
                "confirm it as a static table in the schema if its rows never change",
                label,
                table.name,
            )
            return
        if table.time_leakage_hint and table.time_column_source == "user":
            warning = (
                f"{table.name}.{tcol} is a last-modified time that you confirmed as the event "
                "time: if its rows are rewritten after the events they describe, a feature on "
                "it can use information from after the cutoff"
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
        if table.time_leakage_hint and table.time_column_source != "user":
            self.reject(
                "last_modified_time",
                f"{table.name}.{tcol} is a last-modified time: a row is rewritten after the event "
                "it describes, so even a read bounded by the cutoff shows whether the future "
                "happened (a row updated before the cutoff means it was not updated later). "
                "If the table is an append-only log of events (each change is a new row), set that "
                "column as its event time in the schema; if its rows never change, mark it static",
                label,
                table.name,
            )
            return
        own_terms = [c for c, j in terms if j is None or j == alias]
        date_col = type_kind(next(c.type for c in table.columns if c.name == tcol)) == "date"
        if any(self._reads_future(c, alias, tcol, cl, chain, date_col) for c in own_terms):
            self.reject(
                "reads_future",
                f"{table.name}.{tcol} is selected from after the cutoff: a feature may only read "
                "history, so use '<' the cutoff (the rows after it are the label window)",
                label,
                table.name,
            )
            return
        if any(self._is_bound(c, alias, tcol, cl, chain, date_col) for c in own_terms):
            return
        if not cl:
            self.reject(
                "no_cutoff",
                f"{table.name} is read without any cutoff: the scope has no {LABELS} to bound it by",
                label,
                table.name,
            )
            return
        if not self.allow_rewrite:
            self.reject(
                "no_time_bound",
                f"{table.name}.{tcol} is not bounded by the cutoff in this scope",
                label,
                table.name,
            )
            return
        join = join_of.get(alias)
        ctx = chain[-1]
        position = ctx.order.index(alias) if alias in ctx.order else len(ctx.order)
        bound = None
        for pick in sorted(cl, key=lambda t: (t not in own, t)):
            # an own alias used inside this table's ON must be joined before it
            if (
                join is not None
                and pick in own
                and pick[1] in ctx.order
                and ctx.order.index(pick[1]) >= position
            ):
                continue
            candidate = self._bound_expr(alias, tcol, pick, date_col)
            if (
                self._terminal(_unwrap(candidate.expression, date_ok=True), chain) == pick
            ):  # the name means what we think
                bound = candidate
                break
        if bound is None:
            self.reject(
                "no_time_bound",
                f"{table.name}.{tcol} is not bounded by the cutoff, and a bound cannot be added "
                "safely (no cutoff is reachable by an unshadowed name)",
                label,
                table.name,
            )
            return
        target = f"{alias}.{tcol} < {bound.expression.sql()}"
        if join is not None and join.args.get("on") is not None:
            join.set("on", exp.and_(join.args["on"], bound, copy=False))
        else:
            joins_ok = all(
                str(j.args.get("side") or "").upper() in ("", "INNER")
                for j in select.args.get("joins") or []
            )
            if not joins_ok:
                self.reject(
                    "no_time_bound",
                    f"{table.name}.{tcol} is not bounded by the cutoff, and the scope has outer joins, so the bound cannot be added safely",
                    label,
                    table.name,
                )
                return
            select.set(
                "where",
                exp.Where(this=exp.and_(select.args["where"].this, bound, copy=False))
                if select.args.get("where")
                else exp.Where(this=bound),
            )
        self.rewrites.append(f"added {target} for {table.name} in {label}")

    @staticmethod
    def _bound_expr(alias: str, tcol: str, pick: Terminal, date_col: bool) -> exp.LT:
        k: exp.Expr = exp.column(pick[2], table=pick[1])
        if date_col:  # a date has no time of day: compare whole days, strictly before the cutoff's
            k = exp.Cast(this=k, to=exp.DataType.build("date"))
        return exp.LT(this=exp.column(tcol, table=alias), expression=k)

    def _is_bound(
        self,
        conj: exp.Expr,
        alias: str,
        tcol: str,
        cl: frozenset[Terminal],
        chain: list[_Ctx],
        date_col: bool = False,
    ) -> bool:
        while isinstance(conj, exp.Paren):
            conj = conj.this
        if isinstance(conj, exp.LT):
            time_side, cut_side, strict = conj.this, conj.expression, True
        elif isinstance(conj, exp.GT):
            time_side, cut_side, strict = conj.expression, conj.this, True
        elif isinstance(conj, exp.LTE):
            time_side, cut_side, strict = conj.this, conj.expression, False
        elif isinstance(conj, exp.GTE):
            time_side, cut_side, strict = conj.expression, conj.this, False
        else:
            return False
        t = _unwrap(time_side, date_ok=date_col)
        if not (isinstance(t, exp.Column) and _lc(t.table) == alias and _lc(t.name) == _lc(tcol)):
            return False
        c = _unwrap(cut_side)
        minus_interval = False
        if isinstance(c, exp.Sub) and _positive_interval(c.expression):
            minus_interval = True
            c = _unwrap(c.this)
        if date_col:
            # whole days: the cutoff must be cast to DATE, or same-day events after it pass
            if not isinstance(c, exp.Cast | exp.TryCast):
                return False
            c = _unwrap(c, date_ok=True)
        if not strict and not minus_interval:
            return False  # an event at the cutoff itself belongs to neither history nor window
        term = self._terminal(c, chain)
        return term is not None and term in cl

    def _reads_future(
        self,
        conj: exp.Expr,
        alias: str,
        tcol: str,
        cl: frozenset[Terminal],
        chain: list[_Ctx],
        date_col: bool = False,
    ) -> bool:
        """A term that asks for the table's rows after the cutoff (``t > K``, ``t < K + 30 days``).

        Rewriting such a term to ``t < K`` would be safe but would silently make the feature
        empty; a feature that asks for the future is refused with that reason instead.
        """
        while isinstance(conj, exp.Paren):
            conj = conj.this
        pairs: list[tuple[exp.Expr, str, exp.Expr]] = []  # (time side, op on it, cutoff side)
        if isinstance(conj, exp.LT | exp.LTE):
            pairs.append((conj.this, "upper", conj.expression))
        elif isinstance(conj, exp.GT | exp.GTE):
            pairs.append((conj.this, "lower", conj.expression))
        elif isinstance(conj, exp.Between):
            low, high = conj.args.get("low"), conj.args.get("high")
            if low is not None and high is not None:
                pairs += [(conj.this, "lower", low), (conj.this, "upper", high)]
        flipped = isinstance(conj, exp.LT | exp.LTE | exp.GT | exp.GTE)
        for time_side, kind, cut_side in list(pairs):
            if flipped:  # the column may be on the right: K < t
                t_left = _unwrap(time_side, date_ok=date_col)
                if not (
                    isinstance(t_left, exp.Column)
                    and _lc(t_left.table) == alias
                    and _lc(t_left.name) == _lc(tcol)
                ):
                    pairs.append((cut_side, "upper" if kind == "lower" else "lower", time_side))
        for time_side, kind, cut_side in pairs:
            t = _unwrap(time_side, date_ok=date_col)
            if not (
                isinstance(t, exp.Column) and _lc(t.table) == alias and _lc(t.name) == _lc(tcol)
            ):
                continue
            c = _unwrap(cut_side)
            offset = 0  # +1: cutoff plus an interval, -1: minus, 0: the cutoff itself
            if isinstance(c, exp.Add | exp.Sub) and _positive_interval(c.expression):
                offset = 1 if isinstance(c, exp.Add) else -1
                c = _unwrap(c.this)
            if date_col and isinstance(c, exp.Cast | exp.TryCast):
                c = _unwrap(c, date_ok=True)
            term = self._terminal(c, chain)
            if term is None or term not in cl:
                continue
            if kind == "lower" and offset >= 0:
                return True  # t > K, t >= K, t > K + interval
            if kind == "upper" and offset > 0:
                return True  # t < K + interval
        return False

    def _check_mixing(
        self, select: exp.Select, label: str, cl: frozenset[Terminal], chain: list[_Ctx]
    ) -> None:
        def in_class(node: exp.Expr) -> bool:
            term = self._terminal(node, chain)
            return term is not None and term in cl

        group = select.args.get("group")
        if group is not None:
            if any(group.args.get(k) for k in ("grouping_sets", "rollup", "cube", "totals")):
                self.reject(
                    "aggregate_across_cutoffs",
                    "ROLLUP, CUBE and GROUPING SETS can drop the cutoff from the key",
                    label,
                )
            keys = group.expressions
        else:
            keys = []
        aggregates = [
            a
            for a in select.find_all(exp.AggFunc)
            if self._owner(a, select) and not a.find_ancestor(exp.Window)
        ]
        if (aggregates or group is not None) and not any(in_class(k) for k in keys):
            self.reject(
                "aggregate_across_cutoffs",
                "this aggregate is not grouped by cutoff_time, so it would mix rows of different cutoffs",
                label,
            )
        distinct = select.args.get("distinct")
        if distinct is not None and not any(
            in_class(p.this if isinstance(p, exp.Alias) else p) for p in select.expressions
        ):
            self.reject(
                "distinct_across_cutoffs",
                "DISTINCT without cutoff_time in the output would merge rows of different cutoffs",
                label,
            )
        for window in select.find_all(exp.Window):
            if not self._owner(window, select):
                continue
            if not any(in_class(p) for p in window.args.get("partition_by") or []):
                self.reject(
                    "window_across_cutoffs",
                    "a window function must PARTITION BY the cutoff, or it mixes rows of different cutoffs",
                    label,
                )
        if select.args.get("limit") or select.args.get("offset") or select.args.get("fetch"):
            self.reject(
                "limit_across_cutoffs",
                "LIMIT/OFFSET here would cut across label rows; use it only in a correlated subquery",
                label,
            )

    @staticmethod
    def _owner(node: exp.Expr, select: exp.Select) -> bool:
        """Is ``select`` the innermost SELECT that contains ``node``?"""
        return node.find_ancestor(exp.Select) is select

    # output -------------------------------------------------------------------------------
    def _check_output(self, tree: exp.Select, info: _Info) -> None:
        names = [_lc(p.alias_or_name) for p in tree.expressions]
        if tuple(names) != OUTPUT_COLUMNS:
            raise _Reject(
                Reason(
                    "output_columns",
                    f"the query must return exactly {', '.join(OUTPUT_COLUMNS)} (in that order), not {', '.join(names)}",
                )
            )
        if CUTOFF not in info.cutoff_outputs:
            raise _Reject(
                Reason(
                    "output_cutoff",
                    "cutoff_time in the output must be the cutoff of the labels (l.cutoff_time), not a computed value",
                )
            )


@dataclass
class _Ctx:
    select: exp.Select
    key: int
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> labels | table | derived
    tables: dict[str, Table] = field(default_factory=dict)
    derived: dict[str, _Info] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # FROM source first, then joins


@dataclass
class _Info:
    has_own_cutoff: bool
    cutoff_outputs: dict[str, Terminal]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Terminal, Terminal] = {}

    def __contains__(self, t: Terminal) -> bool:
        return t in self.parent

    def add(self, t: Terminal) -> None:
        self.parent.setdefault(t, t)

    def find(self, t: Terminal) -> Terminal:
        while self.parent[t] != t:
            self.parent[t] = self.parent[self.parent[t]]
            t = self.parent[t]
        return t

    def union(self, a: Terminal, b: Terminal) -> None:
        self.parent[self.find(a)] = self.find(b)

    def same(self, a: Terminal, b: Terminal) -> bool:
        return self.find(a) == self.find(b)

    def single_class(self, members: set[Terminal]) -> bool:
        return len({self.find(t) for t in members}) <= 1


def check(
    sql: str, graph: SchemaGraph, dialect: str = "duckdb", *, allow_rewrite: bool = True
) -> PitResult:
    """Decide whether the feature query ``sql`` can read rows from after the cutoff."""
    return _Checker(graph, dialect, allow_rewrite).run(sql)
