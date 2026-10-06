"""A DuckDB database file, opened read-only with no access to other files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from ml.data.engine import DuckDBSource, EngineError, TableRef, TableSchema
from ml.data.sources.base import ConnectionFailure, ConnectionSpec, Privileges

MAGIC_OFFSET = 8
MAGIC = b"DUCK"


def is_duckdb_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(MAGIC_OFFSET + len(MAGIC))[MAGIC_OFFSET:] == MAGIC
    except OSError:
        return False


class DuckDBFileSource:
    """Wraps ``DuckDBSource(read_only=True)``, which already allows one SELECT only and has
    ``enable_external_access`` off. It is opened per call so no lock is held between them."""

    dialect = "duckdb"

    def __init__(self, spec: ConnectionSpec) -> None:
        self.path = Path(spec.database)

    def _open(self) -> DuckDBSource:
        if not is_duckdb_file(self.path):
            raise ConnectionFailure("not_a_database_file")
        try:
            return DuckDBSource(self.path, read_only=True)
        except duckdb.Error:
            raise ConnectionFailure("not_a_database_file") from None

    def list_tables(self) -> list[TableRef]:
        src = self._open()
        try:
            return src.list_tables()
        finally:
            src.close()

    def table_schema(self, table: TableRef) -> TableSchema:
        src = self._open()
        try:
            return src.table_schema(table)
        finally:
            src.close()

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        src = self._open()
        try:
            return src.query(sql, params, limit=limit, timeout_s=timeout_s)
        except duckdb.Error as e:
            raise EngineError(f"DuckDB rejected the query: {e}") from None
        finally:
            src.close()

    def fingerprint(self) -> str:
        src = self._open()
        try:
            return src.fingerprint()
        finally:
            src.close()

    def check(self) -> None:
        self.list_tables()

    def server_version(self) -> str:
        return f"DuckDB {duckdb.__version__}"

    def privileges(self) -> Privileges:
        return Privileges(can_write=False, notes=["opened read-only (read_only=True)"])
