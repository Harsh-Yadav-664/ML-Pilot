"""A user's DuckDB database file, opened read-only with no access to other files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from ml.data import sql_guard
from ml.data.engine import (
    DUCKDB_COLUMNS_SQL,
    DUCKDB_CONSTRAINTS_SQL,
    DUCKDB_TABLES_SQL,
    INTERNAL_SCHEMA,
    EngineError,
    TableRef,
    TableSchema,
    duckdb_table_schema,
    quote_ident,
)
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
    """A DuckDB file, opened per call (no lock is held between calls) and always closed."""

    dialect = "duckdb"

    def __init__(self, spec: ConnectionSpec) -> None:
        self.path = Path(spec.database)

    # -- for the SQL guard ----------------------------------------------------------

    def connect(self, timeout_s: float = 10) -> duckdb.DuckDBPyConnection:
        """The file opened ``read_only``, unable to read or attach other files, with its
        configuration locked so a query cannot turn that back on."""
        if not is_duckdb_file(self.path):
            raise ConnectionFailure("not_a_database_file")
        try:
            return duckdb.connect(
                str(self.path),
                read_only=True,
                config={
                    "enable_external_access": False,
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                    "lock_configuration": True,
                },
            )
        except duckdb.Error:
            raise ConnectionFailure("not_a_database_file") from None

    def failure(self, exc: BaseException) -> Exception:
        return EngineError(f"DuckDB rejected the query: {str(exc).splitlines()[0]}")

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = sql_guard.catalog(self, DUCKDB_TABLES_SQL, [INTERNAL_SCHEMA])
        return [TableRef(name=name, schema=schema) for schema, name in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        cols = sql_guard.catalog(self, DUCKDB_COLUMNS_SQL, [table.schema, table.name])
        constraints = sql_guard.catalog(self, DUCKDB_CONSTRAINTS_SQL, [table.schema, table.name])
        return duckdb_table_schema(table, cols, constraints)

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        q = sql_guard.guard(sql, self.dialect)
        return sql_guard.execute(self, q, params=params, limit=limit, timeout_s=timeout_s)

    def fingerprint(self) -> str:
        """SHA-256 over every table's name, column types and an order-independent row hash."""
        digest = hashlib.sha256()
        for table in self.list_tables():
            digest.update(f"{table.schema}.{table.name}".encode())
            for col in self.table_schema(table).columns:
                digest.update(f"|{col.name}:{col.type}".encode())
            ((row_hash, n_rows),) = sql_guard.catalog(
                self,
                "SELECT coalesce(bit_xor(hash(t)), 0), count(*) "
                f"FROM {quote_ident(table.schema)}.{quote_ident(table.name)} AS t",
            )
            digest.update(f"|{n_rows}:{row_hash}".encode())
        return digest.hexdigest()

    # -- checks used by the connection manager --------------------------------------

    def check(self) -> None:
        sql_guard.catalog(self, "SELECT 1")

    def server_version(self) -> str:
        return f"DuckDB {duckdb.__version__}"

    def privileges(self) -> Privileges:
        return Privileges(can_write=False, notes=["opened read-only (read_only=True)"])
