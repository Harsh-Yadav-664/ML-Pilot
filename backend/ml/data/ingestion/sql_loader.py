"""SQL dataset loader (interim read-only hardening; the full design is roadmap [2.3])."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlglot import exp

DEFAULT_ROW_LIMIT = 100_000
DEFAULT_TIMEOUT_SECONDS = 30

# Any of these anywhere in the parsed tree (including inside a CTE) is refused.
_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.Command, exp.TruncateTable, exp.Into,
)


class UnsafeQueryError(ValueError):
    """The query is not a single read-only SELECT."""


def _dialect(connection_string: str) -> str | None:
    backend = make_url(connection_string).get_backend_name()
    return {"postgresql": "postgres", "sqlite": "sqlite", "mysql": "mysql", "mssql": "tsql"}.get(backend)


def validate_read_only_query(query: str, dialect: str | None = None) -> exp.Expression:
    """Parse the query and allow exactly one SELECT with no data-modifying parts."""
    try:
        statements = [s for s in sqlglot.parse(query, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeQueryError(f"Could not parse the query: {e}") from e
    if len(statements) != 1:
        raise UnsafeQueryError(f"Exactly one statement is allowed, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeQueryError(f"Data-modifying SQL is not allowed ({node.key.upper()})")
    return tree


def redact(message: str, connection_string: str) -> str:
    """Remove the connection string and its password from a message."""
    out = message.replace(connection_string, "<connection string>")
    try:
        password = make_url(connection_string).password
    except (ArgumentError, ValueError):
        password = None
    if password:
        out = out.replace(str(password), "***")
    return out


class SqlLoader:
    """Load data from a SQL database into a pandas DataFrame, read-only."""

    def load(
        self,
        connection_string: str,
        query: str,
        row_limit: int = DEFAULT_ROW_LIMIT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> pd.DataFrame:
        dialect = _dialect(connection_string)
        tree = validate_read_only_query(query, dialect)
        limited = exp.select("*").from_(tree.subquery("mlpilot_q")).limit(row_limit).sql(dialect=dialect)

        engine = create_engine(connection_string)
        try:
            with engine.connect() as conn, conn.begin() as tx:
                backend = engine.url.get_backend_name()
                if backend == "sqlite":
                    conn.exec_driver_sql("PRAGMA query_only = ON")
                elif backend == "postgresql":
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(timeout_seconds) * 1000}")
                elif backend == "mysql":
                    conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
                    conn.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_seconds) * 1000}")
                df = pd.read_sql_query(text(limited), conn)
                tx.rollback()
        finally:
            engine.dispose()
        return df

    def hash_connection(self, connection_string: str, query: str) -> str:
        """Create a deterministic hash for this query to use as a dataset ID."""
        raw = f"{connection_string}||{query}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
