"""What every live source shares: the connection spec and short error messages.

Queries reach a live source only through ``ml.data.sql_guard``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ml.data.engine import DataSource

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
    def connect(self, timeout_s: float) -> Any:
        """A driver connection, opened read-only where the driver allows (for the SQL guard)."""
        ...

    def failure(self, exc: BaseException) -> Exception:
        """A driver error as MLPilot's own exception, without the driver's text where it
        could carry connection details."""
        ...

    def server_version(self) -> str: ...

    def privileges(self) -> Privileges: ...

    def check(self) -> None:
        """Open a connection and run a trivial query; raises ConnectionFailure."""
        ...


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
