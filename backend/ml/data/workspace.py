"""A project's DuckDB file: where its data versions live as tables.

``data/projects/<project id>/work.duckdb`` holds one table per data version loaded in the
project (named after the file), plus a small registry of which version each table is.
The stored, read-only version files stay the record of what was loaded; the tables are
the working copy every later step reads through ``DataSource``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ml.data.engine import (
    INTERNAL_SCHEMA,
    DuckDBSource,
    EngineError,
    TableRef,
    arrow_to_pandas,
    quote_ident,
    table_name_for,
)
from ml.data.versions import content_hash

REGISTRY = TableRef("registered_versions", INTERNAL_SCHEMA)
VERSION_FILE_SUFFIXES = (".csv", ".parquet")

_OPEN_WORKSPACES: dict[Path, Workspace] = {}
_open_lock = threading.Lock()


class Workspace:
    """One project's DuckDB file. Get it with ``workspace_for``."""

    def __init__(self, source: DuckDBSource) -> None:
        self.source = source
        self._lock = threading.Lock()
        if not source.has_table(REGISTRY):
            source.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(REGISTRY.schema)}")
            source.execute(
                f"CREATE TABLE {REGISTRY.sql()} (version_id VARCHAR PRIMARY KEY, "
                "table_name VARCHAR, n_rows BIGINT, n_columns BIGINT)"
            )

    def table_of(self, version_id: str) -> TableRef | None:
        rows = self.source.execute(
            f"SELECT table_name FROM {REGISTRY.sql()} WHERE version_id = ?", [version_id]
        )
        return TableRef(rows[0][0]) if rows else None

    def register_version(
        self, path: str | Path, version_id: str, filename: str | None = None
    ) -> TableRef:
        """Load the file as a table for this version (once); return the table."""
        with self._lock:
            existing = self.table_of(version_id)
            if existing is not None:
                return existing
            name = table_name_for(filename or Path(path).name)
            if self._table_exists(name):  # same file name, different content
                name = f"{name}_{version_id[:8]}"
            ref = self.source.register_file(path, name)
            self.source.execute(  # registry rows are written by MLPilot, never from user text
                f"INSERT INTO {REGISTRY.sql()} VALUES (?, ?, ?, ?)",
                [
                    version_id,
                    ref.name,
                    self.source.row_count(ref),
                    len(self.source.table_schema(ref).columns),
                ],
            )
            return ref

    def read_version(self, version_id: str) -> pa.Table:
        ref = self.table_of(version_id)
        if ref is None:
            raise EngineError(f"Data version {version_id[:12]} is not loaded in this project")
        return self.source.read_table(ref)

    def load(self, path: str | Path, filename: str | None = None) -> pd.DataFrame:
        """A file as a DataFrame, read through this project's DuckDB table for it.

        A stored version (``<sha256>.csv``) is identified by its name; any other file by
        the SHA-256 of its content, so the same bytes are always the same table.
        """
        path = Path(path)
        stem = path.stem
        version_id = stem if len(stem) == 64 and path.suffix in VERSION_FILE_SUFFIXES else None
        if version_id is None:
            version_id = content_hash(path)
        self.register_version(path, version_id, filename)
        return arrow_to_pandas(self.read_version(version_id))

    def _table_exists(self, name: str) -> bool:
        return self.source.has_table(TableRef(name))


def workspace_for(project_id: str, projects_dir: Path) -> Workspace:
    """The project's workspace, opened once per process (DuckDB allows one instance per file)."""
    if not project_id or any(sep in project_id for sep in ("/", "\\", "..")):
        raise EngineError("Invalid project id")
    path = (projects_dir / project_id / "work.duckdb").resolve()
    with _open_lock:
        ws = _OPEN_WORKSPACES.get(path)
        if ws is None:
            ws = Workspace(DuckDBSource(path))
            _OPEN_WORKSPACES[path] = ws
        return ws


def close_all() -> None:
    """Close every open workspace (tests, shutdown)."""
    with _open_lock:
        for ws in _OPEN_WORKSPACES.values():
            ws.source.close()
        _OPEN_WORKSPACES.clear()
