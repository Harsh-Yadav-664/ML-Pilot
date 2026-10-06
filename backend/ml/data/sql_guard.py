"""The SQL guard: the only way SQL reaches a user's database (#44).

Three independent layers, so a mistake in one does not make a write possible:

1. **Parse** (``guard``): the text is parsed with sqlglot and must be exactly one
   ``SELECT`` (or ``UNION``/``INTERSECT``/``EXCEPT``, with CTEs). Nothing that changes
   data or state may appear anywhere in the tree, CTEs included; side-effect functions
   (``pg_sleep``, ``set_config``, ``lo_import``, ``read_csv``, ...) and sensitive system
   tables are refused; an optional allowlist limits the tables. What runs is the SQL
   re-rendered from the checked tree, so the database executes what was checked.
2. **Session** (``read_only_session``): the database itself refuses writes. Postgres
   runs every statement in ``START TRANSACTION READ ONLY`` with ``statement_timeout`` and
   ``idle_in_transaction_session_timeout``, always rolled back. SQLite is opened
   ``mode=ro`` with ``query_only`` and an authorizer that allows reads only. DuckDB is
   opened ``read_only`` without access to other files, and runs one statement at a time.
3. **Privileges**: the connection test reports ``can_write`` when the role could write
   anything (``SourceWithChecks.privileges``), so a user can see that the role is not
   read-only even though layers 1 and 2 stop the writes.

Limits on every query: a row limit (the query is wrapped and more rows raise, never
truncate), a result-size limit and a time limit. Each executed query is logged with the
SHA-256 of its text, its duration and row count, never its text or parameters.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import duckdb
import pg8000.exceptions
import pyarrow as pa
import sqlglot
from sqlglot import exp

from ml.data.engine import EngineError, QueryRejected, QueryTimeout, RowLimitExceeded

logger = logging.getLogger("mlpilot.sql")
# sqlglot's own warnings quote the SQL text, which can hold data; keep them out of the logs.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 100_000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
FETCH_BATCH = 10_000
CATALOG_TIMEOUT_S = 15
DIALECTS = ("postgres", "sqlite", "duckdb")

# Functions with side effects, or that reach outside the database (files, other servers,
# sessions, settings). Matched on the bare name, lower case.
DENIED_FUNCTIONS = frozenset(
    {
        # Postgres
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "set_config",
        "nextval",
        "setval",
        "currval",
        "lastval",
        "pg_notify",
        "pg_stat_file",
        "pg_current_logfile",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "query_to_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "table_to_xml",
        "table_to_xml_and_xmlschema",
        "schema_to_xml",
        "schema_to_xml_and_xmlschema",
        "database_to_xml",
        "database_to_xml_and_xmlschema",
        "txid_current",
        "pg_current_xact_id",
        "pg_export_snapshot",
        "pg_import_system_collations",
        "pg_switch_wal",
        "pg_promote",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_log_backend_memory_contexts",
        "make_lo",
        # SQLite
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        # DuckDB
        "glob",
        "getenv",
        "query",
        "query_table",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "parquet_kv_metadata",
        "parquet_file_metadata",
        "sniff_csv",
        "iceberg_scan",
        "delta_scan",
        "sqlite_scan",
        "postgres_scan",
        "postgres_query",
        "mysql_query",
        "install",
        "load",
    }
)
DENIED_PREFIXES = (
    "dblink",
    "lo_",
    "pg_read_",
    "pg_ls_",
    "pg_file",
    "pg_advisory",
    "pg_try_advisory",
    "pg_terminate",
    "pg_cancel",
    "pg_create_",
    "pg_drop_replication",
    "pg_replication_",
    "pg_logical_",
    "pg_backup",
    "pg_start_backup",
    "pg_stop_backup",
    "pg_wal_replay",
    "pg_stat_reset",
    "pg_log_",
    "read_",  # DuckDB file readers (read_csv, read_parquet, read_text, ...)
)
# System tables that hold secrets or other sessions' work, refused even without an allowlist.
DENIED_TABLES = frozenset(
    {
        "pg_authid",
        "pg_shadow",
        "pg_user_mapping",
        "pg_user_mappings",
        "pg_largeobject",
        "pg_largeobject_metadata",
        "pg_stat_activity",
        "pg_hba_file_rules",
        "pg_ident_file_mappings",
        "pg_file_settings",
        "pg_subscription",
        "duckdb_secrets",
    }
)


def _classes(*names: str) -> tuple[type[exp.Expression], ...]:
    """The sqlglot classes of these names that exist in the installed version."""
    return tuple(getattr(exp, n) for n in names if isinstance(getattr(exp, n, None), type))


# Node types that may not appear anywhere in a query, with the rejection code for each.
# Checked before the statement type, so a data-modifying CTE is reported as what it is.
FORBIDDEN_NODES: list[tuple[tuple[type[exp.Expression], ...], str]] = [
    (_classes("Insert", "Update", "Delete", "Merge"), "dml"),
    (
        _classes(
            "Create",
            "Drop",
            "Alter",
            "TruncateTable",
            "Grant",
            "Revoke",
            "Comment",
            "Refresh",
            "Analyze",
            "LoadData",
            "Cache",
            "Uncache",
            "Install",
        ),
        "ddl",
    ),
    (_classes("Into"), "select_into"),
    (_classes("Lock"), "locking_clause"),
    (_classes("Copy"), "copy"),
    (_classes("Set", "SetItem"), "set"),
    (_classes("Transaction", "Commit", "Rollback"), "transaction_control"),
    (_classes("Command"), "command"),  # CALL, DO, EXPLAIN, PREPARE, LOCK, VACUUM, SHOW, ...
    (_classes("Pragma", "Attach", "Detach", "Use", "Kill", "Describe", "Show"), "not_a_select"),
]
REASONS = {
    "dml": "it changes data (also refused inside a CTE)",
    "ddl": "it changes the schema or permissions",
    "select_into": "SELECT ... INTO creates a table",
    "locking_clause": "it locks rows (FOR UPDATE / FOR SHARE)",
    "copy": "COPY reads or writes files and programs",
    "set": "it changes settings",
    "transaction_control": "it controls the transaction",
    "command": "it is a command, not a query",
    "not_a_select": "it is not a query",
}
ALLOWED_ROOTS = _classes("Select", "Union", "Intersect", "Except")


class SqlRejected(QueryRejected):
    """Layer 1 refused the SQL. ``code`` is stable (tests and the UI use it)."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{reason} [{code}]")


class ResultTooLarge(EngineError):
    """The result is bigger than the allowed number of bytes."""


@dataclass(frozen=True)
class GuardedQuery:
    """SQL that passed layer 1. Only ``guard`` builds these (a test checks)."""

    sql: str  # re-rendered from the checked tree
    dialect: str
    tree: exp.Query
    tables: frozenset[str]
    sha256: str  # of the text as given, for the query log


class GuardedSource(Protocol):
    """What a live source gives the guard: a connection opened read-only where the driver
    allows it, and a way to turn driver errors into MLPilot's short messages."""

    dialect: str

    def connect(self, timeout_s: float) -> Any: ...

    def failure(self, exc: BaseException) -> Exception: ...


# -- layer 1 -----------------------------------------------------------------------------


def guard(sql: str, dialect: str, *, allowed_tables: set[str] | None = None) -> GuardedQuery:
    """Check ``sql`` (layer 1). Raises ``SqlRejected`` with a code, or returns the query.

    ``allowed_tables`` holds lower-case ``name`` or ``schema.name`` entries (from schema
    introspection); when given, every table the query reads must be one of them.
    """
    if dialect not in DIALECTS:
        raise SqlRejected("unsupported_dialect", f"Unsupported dialect {dialect!r}")
    digest = hashlib.sha256(sql.encode()).hexdigest()
    try:
        tree = _check(sql, dialect, allowed_tables)
    except SqlRejected as e:
        logger.info("sql rejected sha256=%s dialect=%s code=%s", digest[:16], dialect, e.code)
        raise
    return GuardedQuery(
        sql=tree.sql(dialect=dialect, comments=False),
        dialect=dialect,
        tree=tree,
        tables=frozenset(_table_names(tree, dialect)),
        sha256=digest,
    )


def _check(sql: str, dialect: str, allowed_tables: set[str] | None) -> exp.Query:
    if not sql.strip():
        raise SqlRejected("empty", "The query is empty")
    if len(sql) > MAX_SQL_CHARS:
        raise SqlRejected("too_long", f"The query is longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise SqlRejected("parse_error", f"Could not parse the query: {_first_line(e)}") from None
    if len(statements) != 1:
        raise SqlRejected(
            "multi_statement", f"Exactly one statement is allowed, got {len(statements)}"
        )
    tree = statements[0]
    for node in tree.walk():
        for classes, code in FORBIDDEN_NODES:
            if isinstance(node, classes):
                raise SqlRejected(
                    code,
                    f"Only read-only SELECT queries can run; {node.key.upper()}: {REASONS[code]}",
                )
    if not isinstance(tree, ALLOWED_ROOTS):
        raise SqlRejected(
            "not_a_select", f"Only SELECT queries are allowed, got {tree.key.upper()}"
        )
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in DENIED_FUNCTIONS or name.startswith(DENIED_PREFIXES):
            raise SqlRejected("denied_function", f"The function {name}() is not allowed")
    names = _table_names(tree, dialect)
    for name in names:
        if name.rsplit(".", 1)[-1] in DENIED_TABLES:
            raise SqlRejected("denied_table", f"The system table {name} is not allowed")
    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables}
        for name in names:
            if name not in allowed:
                raise SqlRejected("table_not_allowed", f"The table {name} is not in the schema")
    assert isinstance(tree, exp.Query)
    return tree


def _first_line(e: Exception) -> str:
    return str(e).splitlines()[0][:200] if str(e) else type(e).__name__


def _function_name(func: exp.Func) -> str:
    if isinstance(func, exp.Anonymous):
        return str(func.name).lower()
    return func.sql_name().lower()


def _ident(identifier: exp.Expression | None, dialect: str) -> str:
    if not isinstance(identifier, exp.Identifier):
        return identifier.name.lower() if identifier is not None else ""
    # Postgres folds unquoted names to lower case and keeps quoted ones; SQLite and DuckDB
    # match names case-insensitively. Lower case is the common key for the allowlist.
    return str(identifier.this).lower()


def _table_names(tree: Any, dialect: str) -> set[str]:
    """Every table the query reads, as ``name`` or ``schema.name`` (CTE names excluded)."""
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    names: set[str] = set()
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # a table function such as generate_series(...); functions are checked
        name = _ident(table.this, dialect)
        schema = _ident(table.args.get("db"), dialect)
        catalog = _ident(table.args.get("catalog"), dialect)
        if not schema and name in ctes:
            continue
        parts = [p for p in (catalog, schema, name) if p]
        names.add(".".join(parts))
    return names


def allowed_tables(source: Any) -> set[str]:
    """The allowlist for ``guard``: every introspected table as ``name`` and ``schema.name``."""
    names: set[str] = set()
    for table in source.list_tables():
        names.add(table.name.lower())
        names.add(f"{table.schema}.{table.name}".lower())
    return names


# -- layer 2 and the limits ------------------------------------------------------------


def execute(
    source: GuardedSource,
    q: GuardedQuery,
    *,
    limit: int,
    timeout_s: float,
    params: Sequence[Any] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> pa.Table:
    """Run a guarded query read-only. More than ``limit`` rows, ``max_bytes`` bytes or
    ``timeout_s`` seconds raise (``RowLimitExceeded``, ``ResultTooLarge``, ``QueryTimeout``)."""
    if not isinstance(q, GuardedQuery):
        raise TypeError("execute() takes the result of guard(), not SQL text")
    if q.dialect != source.dialect:
        raise QueryRejected(f"The query was checked for {q.dialect}, not {source.dialect}")
    if params and source.dialect == "postgres":
        raise EngineError("Query parameters are not supported for Postgres sources")
    wrapped = exp.select("*").from_(q.tree.subquery("mlpilot_q")).limit(limit + 1)
    sql = wrapped.sql(dialect=q.dialect, comments=False)
    start = time.monotonic()
    outcome = "failed"
    rows = 0
    nbytes = 0
    try:
        with read_only_session(source, timeout_s) as session:
            table = session.fetch_arrow(
                sql, list(params or []), max_rows=limit, max_bytes=max_bytes
            )
        rows, nbytes = table.num_rows, table.nbytes
        outcome = "ok"
        return table
    except RowLimitExceeded:
        outcome = "row_limit"
        raise
    except ResultTooLarge:
        outcome = "too_large"
        raise
    except QueryTimeout:
        outcome = "timeout"
        raise
    finally:
        logger.info(
            "sql executed sha256=%s dialect=%s outcome=%s rows=%d bytes=%d duration_ms=%d",
            q.sha256[:16],
            q.dialect,
            outcome,
            rows,
            nbytes,
            (time.monotonic() - start) * 1000,
        )


def catalog(
    source: GuardedSource,
    sql: str,
    params: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    timeout_s: float = CATALOG_TIMEOUT_S,
) -> list[tuple[Any, ...]]:
    """One of MLPilot's own fixed introspection queries (tables, columns, privileges).

    It passes the same guard and runs in the same read-only session as a user's query.
    Postgres parameters are named (``:name``) and bound as literals into the checked tree;
    SQLite and DuckDB take ``?`` parameters.
    """
    q = guard(sql, source.dialect)
    text = q.sql
    args: list[Any] = []
    if isinstance(params, Mapping):
        text = _bind_named(q.tree, params).sql(dialect=q.dialect, comments=False)
    elif params:
        args = list(params)
    with read_only_session(source, timeout_s) as session:
        _, rows = session.fetch_rows(text, args, max_rows=None, max_bytes=DEFAULT_MAX_BYTES)
    return rows


def _bind_named(tree: exp.Query, params: Mapping[str, Any]) -> exp.Expression:
    def bind(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Placeholder) and node.name in params:
            value = params[node.name]
            if isinstance(value, bool) or value is None:
                raise EngineError("Only text and number parameters are supported")
            if isinstance(value, int | float):
                return exp.Literal.number(value)
            return exp.Literal.string(str(value))
        return node

    return tree.copy().transform(bind)


@contextmanager
def read_only_session(source: GuardedSource, timeout_s: float) -> Iterator[_Session]:
    """Layer 2: a session in which the database itself refuses writes. Always closed."""
    handle = source.connect(timeout_s)
    session: _Session
    if source.dialect == "postgres":
        session = _PostgresSession(source, handle, timeout_s)
    elif source.dialect == "sqlite":
        session = _SqliteSession(source, handle, timeout_s)
    elif source.dialect == "duckdb":
        session = _DuckDBSession(source, handle, timeout_s)
    else:
        handle.close()
        raise EngineError(f"Unsupported dialect {source.dialect!r}")
    try:
        session.begin()
        yield session
    finally:
        session.end()


class _Session:
    """The one place statements are executed on a user's database."""

    def __init__(self, source: GuardedSource, handle: Any, timeout_s: float) -> None:
        self.source = source
        self.handle = handle
        self.timeout_s = timeout_s

    def begin(self) -> None: ...

    def end(self) -> None:
        self.handle.close()

    def fetch_rows(
        self, sql: str, params: list[Any], *, max_rows: int | None, max_bytes: int
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        raise NotImplementedError

    def fetch_arrow(
        self, sql: str, params: list[Any], *, max_rows: int, max_bytes: int
    ) -> pa.Table:
        names, rows = self.fetch_rows(sql, params, max_rows=max_rows, max_bytes=max_bytes)
        table = rows_to_arrow(names, rows)
        _check_bytes(table.nbytes, max_bytes)
        return table


_PG_ERRORS = (pg8000.exceptions.Error, OSError)


class _PostgresSession(_Session):
    def begin(self) -> None:
        ms = int(self.timeout_s * 1000)
        try:
            self.handle.run("START TRANSACTION READ ONLY")
            self.handle.run(f"SET LOCAL statement_timeout = {ms}")
            self.handle.run(f"SET LOCAL idle_in_transaction_session_timeout = {ms + 5000}")
        except _PG_ERRORS as e:
            self.handle.close()
            raise self.source.failure(e) from None

    def end(self) -> None:
        try:
            self.handle.run("ROLLBACK")
        except Exception:  # noqa: BLE001 - the connection is closed next either way
            logger.debug("rollback failed; closing the connection")
        finally:
            self.handle.close()

    def fetch_rows(
        self, sql: str, params: list[Any], *, max_rows: int | None, max_bytes: int
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        if params:
            raise EngineError("Positional parameters are not supported for Postgres")
        try:
            # The extended protocol, not pg8000's default simple one for parameterless SQL:
            # the server then accepts exactly one statement, so "ROLLBACK; DELETE ..." can't
            # end the read-only transaction and run a write after it.
            context = self.handle.execute_unnamed(sql)
        except _PG_ERRORS as e:
            raise self.source.failure(e) from None
        names = [c["name"] for c in context.columns or []]
        rows = [tuple(r) for r in context.rows or []]
        _check_rows(len(rows), max_rows)
        return names, rows


# SQLite's own side-effect functions (the GLOB operator calls glob(), so it stays allowed).
SQLITE_DENIED_FUNCTIONS = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
)
SQLITE_READ_PRAGMAS = frozenset(
    {"table_info", "table_xinfo", "index_list", "index_info", "foreign_key_list"}
)


class _SqliteSession(_Session):
    def begin(self) -> None:
        con: sqlite3.Connection = self.handle
        deadline = time.monotonic() + self.timeout_s
        try:
            con.execute("PRAGMA query_only = ON")
            con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            con.set_authorizer(_sqlite_authorizer)
            con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        except sqlite3.Error as e:
            con.close()
            raise self.source.failure(e) from None

    def fetch_rows(
        self, sql: str, params: list[Any], *, max_rows: int | None, max_bytes: int
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        try:
            cur = self.handle.execute(sql, params)
            names = [d[0] for d in cur.description or []]
            rows: list[tuple[Any, ...]] = []
            while batch := cur.fetchmany(FETCH_BATCH):
                rows.extend(batch)
                _check_rows(len(rows), max_rows)
            return names, rows
        except sqlite3.Error as e:
            raise self.source.failure(e) from None


def _sqlite_authorizer(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
    """Allow reading only: SELECT, column reads, functions not on the denylist, and the
    read-only table-valued pragmas. Everything else (writes, ATTACH, ...) is refused."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        denied = (arg2 or "").lower() in SQLITE_DENIED_FUNCTIONS
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in SQLITE_READ_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class _DuckDBSession(_Session):
    def begin(self) -> None:
        self._timer = threading.Timer(self.timeout_s, self.handle.interrupt)
        self._timer.start()

    def end(self) -> None:
        self._timer.cancel()
        self.handle.close()

    def fetch_rows(
        self, sql: str, params: list[Any], *, max_rows: int | None, max_bytes: int
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        table = self._run(sql, params, max_rows, max_bytes)
        return table.column_names, [tuple(r.values()) for r in table.to_pylist()]

    def fetch_arrow(
        self, sql: str, params: list[Any], *, max_rows: int, max_bytes: int
    ) -> pa.Table:
        return self._run(sql, params, max_rows, max_bytes)

    def _run(self, sql: str, params: list[Any], max_rows: int | None, max_bytes: int) -> pa.Table:
        try:
            # DuckDB runs every statement in a string; allow exactly one, whatever layer 1 said.
            if len(duckdb.extract_statements(sql)) != 1:
                raise QueryRejected("Exactly one statement is allowed")
            reader = self.handle.execute(sql, params).to_arrow_reader(FETCH_BATCH)
            batches: list[pa.RecordBatch] = []
            rows = nbytes = 0
            for batch in reader:
                rows += batch.num_rows
                nbytes += batch.nbytes
                _check_rows(rows, max_rows)
                _check_bytes(nbytes, max_bytes)
                batches.append(batch)
            return pa.Table.from_batches(batches, schema=reader.schema)
        except duckdb.InterruptException:
            raise QueryTimeout(
                f"The query ran longer than {self.timeout_s:g} s and was stopped"
            ) from None
        except duckdb.Error as e:
            raise self.source.failure(e) from None


def _check_rows(rows: int, max_rows: int | None) -> None:
    if max_rows is not None and rows > max_rows:
        raise RowLimitExceeded(f"The query returned more than {max_rows} rows")


def _check_bytes(nbytes: int, max_bytes: int) -> None:
    if nbytes > max_bytes:
        raise ResultTooLarge(f"The result is larger than {max_bytes // (1024 * 1024)} MiB")


def rows_to_arrow(names: list[str], rows: list[tuple[Any, ...]]) -> pa.Table:
    """Rows from a driver as Arrow; a column Arrow can't type is kept as text."""
    columns: dict[str, pa.Array] = {}
    for i, name in enumerate(names):
        values = [r[i] for r in rows]
        values = [float(v) if isinstance(v, Decimal) else v for v in values]
        try:
            columns[name] = pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            columns[name] = pa.array([None if v is None else str(v) for v in values], pa.string())
    return pa.table(columns)
