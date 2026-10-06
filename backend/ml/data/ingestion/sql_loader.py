"""The old SQL import: a connection string and a query in one request (until the UI uses
saved connections, #46). The string is turned into a ``ConnectionSpec`` and the query runs
through the same source and SQL guard as a saved connection."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from ml.data.engine import EngineError, arrow_to_pandas
from ml.data.sources import ConnectionSpec, open_source

DEFAULT_ROW_LIMIT = 100_000
DEFAULT_TIMEOUT_SECONDS = 30
_BACKENDS = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "sqlite": "sqlite",
    "duckdb": "duckdb",
}


def spec_from_url(connection_string: str) -> ConnectionSpec:
    """``postgresql://user:pw@host:port/db``, ``sqlite:///path`` or ``duckdb:///path``."""
    try:
        url = make_url(connection_string)
    except (ArgumentError, ValueError):
        raise EngineError("The connection string could not be read") from None
    dialect = _BACKENDS.get(url.get_backend_name())
    if dialect is None:
        raise EngineError("Only postgresql://, sqlite:/// and duckdb:/// connections are supported")
    if dialect != "postgres":
        if not url.database or url.database == ":memory:":
            raise EngineError(f"A {dialect} connection needs the path of a database file")
        return ConnectionSpec(dialect, url.database)  # type: ignore[arg-type]
    return ConnectionSpec(
        "postgres",
        url.database or "postgres",
        host=url.host,
        port=url.port,
        username=url.username,
        password=url.password if url.password is None else str(url.password),
        ssl_mode=str(url.query.get("sslmode", "prefer")),
    )


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
    """Load the result of one read-only query into a DataFrame."""

    def load(
        self,
        connection_string: str,
        query: str,
        row_limit: int = DEFAULT_ROW_LIMIT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if kwargs:
            raise TypeError(f"Unknown options: {sorted(kwargs)}")
        source = open_source(spec_from_url(connection_string))
        return arrow_to_pandas(source.query(query, limit=row_limit, timeout_s=timeout_seconds))

    def hash_connection(self, connection_string: str, query: str) -> str:
        """Create a deterministic hash for this query to use as a dataset ID."""
        raw = f"{connection_string}||{query}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
