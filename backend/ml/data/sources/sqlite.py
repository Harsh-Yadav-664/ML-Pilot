"""SQLite through the standard library, opened ``mode=ro``; queries go through the SQL guard."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow as pa

from ml.data import sql_guard
from ml.data.engine import ColumnInfo, EngineError, QueryTimeout, TableRef, TableSchema
from ml.data.sources.base import ConnectionFailure, ConnectionSpec, Privileges

HEADER = b"SQLite format 3\x00"


def is_sqlite_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(HEADER)) == HEADER
    except OSError:
        return False


class SqliteSource:
    """A SQLite file, read-only. The file is opened per call and always closed."""

    dialect = "sqlite"

    def __init__(self, spec: ConnectionSpec) -> None:
        self.path = Path(spec.database)

    # -- for the SQL guard ----------------------------------------------------------

    def connect(self, timeout_s: float = 10) -> sqlite3.Connection:
        """The file opened read-only (``mode=ro``); the guard adds the rest of layer 2."""
        if not is_sqlite_file(self.path):
            raise ConnectionFailure("not_a_database_file")
        uri = f"file:{quote(str(self.path.resolve()))}?mode=ro"
        try:
            return sqlite3.connect(uri, uri=True, timeout=timeout_s)
        except sqlite3.Error:
            raise ConnectionFailure("not_a_database_file") from None

    def failure(self, exc: BaseException) -> Exception:
        text = str(exc)
        if "interrupt" in text.lower():
            return QueryTimeout("The query ran past its time limit and was stopped")
        if isinstance(exc, sqlite3.DatabaseError) and "file is not a database" in text:
            return ConnectionFailure("not_a_database_file")
        return EngineError(f"SQLite rejected the query: {text}")

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = sql_guard.catalog(
            self,
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        return [TableRef(name=r[0], schema="main") for r in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        info = sql_guard.catalog(
            self,
            'SELECT cid, name, type, "notnull", dflt_value, pk FROM pragma_table_info(?)',
            [table.name],
        )
        if not info:
            raise EngineError(f"Table {table.name} does not exist")
        pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
        return TableSchema(
            table=table,
            columns=[ColumnInfo(r[1], r[2] or "", not r[3]) for r in info],
            primary_key=pk,
        )

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        """One SELECT, guarded and read-only. Raises on anything else, a timeout or > limit rows."""
        q = sql_guard.guard(sql, self.dialect)
        return sql_guard.execute(self, q, params=params, limit=limit, timeout_s=timeout_s)

    def fingerprint(self) -> str:
        """SHA-256 of the file's bytes (the whole database, since it is one file)."""
        digest = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # -- checks used by the connection manager --------------------------------------

    def check(self) -> None:
        sql_guard.catalog(self, "SELECT 1")

    def server_version(self) -> str:
        return f"SQLite {sqlite3.sqlite_version}"

    def privileges(self) -> Privileges:
        return Privileges(can_write=False, notes=["opened read-only (mode=ro, query_only)"])
