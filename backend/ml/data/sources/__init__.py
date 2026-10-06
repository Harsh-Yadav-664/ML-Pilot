"""Live database sources: Postgres, SQLite and DuckDB files, all read-only.

Each source implements ``ml.data.engine.DataSource``. ``open_source`` builds one from a
``ConnectionSpec`` (the resolved settings of a saved connection, password included).
"""

from __future__ import annotations

from ml.data.sources.base import (
    ConnectionFailure,
    ConnectionSpec,
    Privileges,
    SourceWithChecks,
    open_source,
)

__all__ = [
    "ConnectionFailure",
    "ConnectionSpec",
    "Privileges",
    "SourceWithChecks",
    "open_source",
]
