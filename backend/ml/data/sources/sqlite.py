"""SQLite through the standard library, opened ``mode=ro`` with ``query_only`` on."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow as pa

from ml.data.engine import (
    ColumnInfo,
    EngineError,
    QueryTimeout,
    RowLimitExceeded,
    TableRef,
    TableSchema,
)
from ml.data.sources.base import (
    ConnectionFailure,
    ConnectionSpec,
    Deadline,
    Privileges,
    guard,
    limited_sql,
    rows_to_arrow,
)

HEADER = b"SQLite format 3\x00"
PROGRESS_STEPS = 10_000  # VM instructions between deadline checks


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

    def _open(self, timeout_s: float = 10) -> sqlite3.Connection:
        if not is_sqlite_file(self.path):
            raise ConnectionFailure("not_a_database_file")
        uri = f"file:{quote(str(self.path.resolve()))}?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True, timeout=timeout_s)
            con.execute("PRAGMA query_only = ON")
            return con
        except sqlite3.Error:
            raise ConnectionFailure("not_a_database_file") from None

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = self._rows(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [TableRef(name=r[0], schema="main") for r in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        quoted = '"' + table.name.replace('"', '""') + '"'
        info = self._rows(f"PRAGMA table_info({quoted})")
        if not info:
            raise EngineError(f"Table {table.name} does not exist")
        # cid, name, type, notnull, dflt_value, pk
        pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
        return TableSchema(
            table=table,
            columns=[ColumnInfo(r[1], r[2] or "", not r[3]) for r in info],
            primary_key=pk,
        )

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        """One SELECT on a read-only handle. Raises on anything else, a timeout or > limit rows."""
        guard(sql, "sqlite")
        wrapped = limited_sql(sql, "sqlite", limit)
        deadline = Deadline(timeout_s)
        con = self._open(timeout_s)
        try:
            con.set_progress_handler(lambda: 1 if deadline.expired() else 0, PROGRESS_STEPS)
            cur = con.execute(wrapped, params or [])
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        except sqlite3.OperationalError as e:
            if "interrupt" in str(e).lower():
                raise QueryTimeout(
                    f"The query ran longer than {timeout_s} s and was stopped"
                ) from None
            raise EngineError(f"SQLite rejected the query: {e}") from None
        finally:
            con.close()
        if len(rows) > limit:
            raise RowLimitExceeded(f"The query returned more than {limit} rows")
        return rows_to_arrow(names, rows)

    def fingerprint(self) -> str:
        """SHA-256 of the file's bytes (the whole database, since it is one file)."""
        digest = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # -- checks used by the connection manager --------------------------------------

    def check(self) -> None:
        self._rows("SELECT 1")

    def server_version(self) -> str:
        return f"SQLite {sqlite3.sqlite_version}"

    def privileges(self) -> Privileges:
        return Privileges(can_write=False, notes=["opened read-only (mode=ro, query_only)"])

    def _rows(self, sql: str) -> list[tuple]:
        con = self._open()
        try:
            return con.execute(sql).fetchall()
        except sqlite3.Error:
            raise ConnectionFailure("not_a_database_file") from None
        finally:
            con.close()
