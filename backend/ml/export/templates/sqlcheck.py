"""The SQL guard rules of MLPilot, as far as this bundle needs them: one SELECT, nothing else.

A vendored copy of the rules in MLPilot's ml/data/sql_guard.py (layer 1: single statement, parsed
with sqlglot, a query only, no data or schema changes anywhere inside it, no file or session
functions, no system tables holding secrets). tests/unit/test_export_sqlcheck.py runs the same
hostile matrix against it and fails if the copied rules drift. score.py also opens its
connection read-only; this check is the second layer and refuses before anything is sent.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

MAX_SQL_CHARS = 100_000_000  # the VALUES list of a validation set is long; the rules are not

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


class SqlRefused(ValueError):
    """The statement is not a single read-only SELECT; ``code`` is the guard's code."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        super().__init__(f"{code}: {reason}")


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


def check(sql: str, dialect: str) -> exp.Query:
    if not sql.strip():
        raise SqlRefused("empty", "The query is empty")
    if len(sql) > MAX_SQL_CHARS:
        raise SqlRefused("too_long", f"The query is longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise SqlRefused("parse_error", f"Could not parse the query: {_first_line(e)}") from None
    if len(statements) != 1:
        raise SqlRefused(
            "multi_statement", f"Exactly one statement is allowed, got {len(statements)}"
        )
    tree = statements[0]
    for node in tree.walk():
        for classes, code in FORBIDDEN_NODES:
            if isinstance(node, classes):
                raise SqlRefused(
                    code,
                    f"Only read-only SELECT queries can run; {node.key.upper()}: {REASONS[code]}",
                )
    if not isinstance(tree, ALLOWED_ROOTS):
        raise SqlRefused("not_a_select", f"Only SELECT queries are allowed, got {tree.key.upper()}")
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in DENIED_FUNCTIONS or name.startswith(DENIED_PREFIXES):
            raise SqlRefused("denied_function", f"The function {name}() is not allowed")
    names = _table_names(tree, dialect)
    for name in names:
        if name.rsplit(".", 1)[-1] in DENIED_TABLES:
            raise SqlRefused("denied_table", f"The system table {name} is not allowed")
    assert isinstance(tree, exp.Query)
    return tree
