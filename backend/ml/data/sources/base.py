"""What every live source shares: the connection spec, short error messages, the guard hook."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import pyarrow as pa

from ml.data.engine import DataSource, QueryRejected
from ml.data.ingestion.sql_loader import UnsafeQueryError, validate_read_only_query

Dialect = Literal["postgres", "sqlite", "duckdb"]
SSL_MODES = ("disable", "prefer", "require", "verify-ca", "verify-full")

# Short, fixed messages. A driver's own error text can contain host names, user names or
# parts of the connection string, so it is never passed on; only one of these is.
MESSAGES = {
    "auth_failed": "Authentication failed: check the user name and password.",
    "host_unreachable": "Could not reach the database: check the host, the port and the network.",
    "timeout": "The connection or the query timed out.",
    "ssl_required": "The server requires SSL: set ssl_mode to require or verify-full.",
    "ssl_failed": "The SSL handshake failed: check ssl_mode and the root certificate.",
    "database_not_found": "The database does not exist or this user cannot open it.",
    "not_a_database_file": "That path is not a readable database file of the chosen type.",
    "secret_unavailable": "The saved password could not be read (see the secret setup).",
    "other": "The database refused the request.",
}


class ConnectionFailure(Exception):
    """A connection or query failed. ``code`` is a key of MESSAGES; the text never has secrets."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code if code in MESSAGES else "other"
        self.message = MESSAGES[self.code] + (f" ({detail})" if detail else "")
        super().__init__(self.message)


@dataclass(frozen=True)
class ConnectionSpec:
    """A saved connection with its secret resolved. Never log or return this object."""

    dialect: Dialect
    database: str  # database name, or the file path for sqlite and duckdb
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    ssl_mode: str = "prefer"
    ssl_root_cert: str | None = None


@dataclass(frozen=True)
class Privileges:
    """What the connected role could do. ``can_write`` is the input to the read-only layers (#44)."""

    can_write: bool
    notes: list[str]


class SourceWithChecks(DataSource, Protocol):
    def server_version(self) -> str: ...

    def privileges(self) -> Privileges: ...

    def check(self) -> None:
        """Open a connection and run a trivial query; raises ConnectionFailure."""
        ...


def guard(sql: str, dialect: str) -> None:
    """Layer 1: one SELECT, nothing that modifies data (shared with the SQL import)."""
    try:
        validate_read_only_query(sql, dialect)
    except UnsafeQueryError as e:
        raise QueryRejected(str(e)) from e


def limited_sql(sql: str, dialect: str, limit: int) -> str:
    """The query wrapped to fetch at most ``limit + 1`` rows, so overflow can be detected."""
    from sqlglot import exp

    tree = validate_read_only_query(sql, dialect)
    return exp.select("*").from_(tree.subquery("mlpilot_q")).limit(limit + 1).sql(dialect=dialect)


def rows_to_arrow(names: list[str], rows: list[tuple]) -> pa.Table:
    """Rows from a DB-API cursor as Arrow; a column Arrow can't type is kept as text."""
    from decimal import Decimal

    columns: dict[str, pa.Array] = {}
    for i, name in enumerate(names):
        values = [r[i] for r in rows]
        values = [float(v) if isinstance(v, Decimal) else v for v in values]
        try:
            columns[name] = pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            columns[name] = pa.array([None if v is None else str(v) for v in values], pa.string())
    return pa.table(columns)


class Deadline:
    """A wall-clock limit for a query, polled by drivers that have no server-side timeout."""

    def __init__(self, seconds: float) -> None:
        self.at = time.monotonic() + seconds

    def expired(self) -> bool:
        return time.monotonic() > self.at


def open_source(spec: ConnectionSpec) -> SourceWithChecks:
    if spec.dialect == "postgres":
        from ml.data.sources.postgres import PostgresSource

        return PostgresSource(spec)
    if spec.dialect == "sqlite":
        from ml.data.sources.sqlite import SqliteSource

        return SqliteSource(spec)
    if spec.dialect == "duckdb":
        from ml.data.sources.duckdb_file import DuckDBFileSource

        return DuckDBFileSource(spec)
    raise ConnectionFailure("other", f"unsupported dialect {spec.dialect!r}")
