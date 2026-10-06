"""The one engine MLPilot reads data with: DuckDB behind a small ``DataSource`` interface.

Uploaded CSV/Parquet files become tables in a per-project DuckDB file, so a file is just
a one-table database and every later step (profiling, leakage checks, training) reads
through the same interface. Results leave as Arrow and become pandas only at the model
boundary (``arrow_to_pandas``).

Two connections are kept apart on purpose:

* ``read_file`` runs on a throw-away in-memory connection, because only that one may
  open files on disk (``read_csv`` / ``read_parquet``).
* The project's ``work.duckdb`` is opened with ``enable_external_access=false``. It can
  neither read nor write other files, attach other databases or load extensions, so a
  query that reaches it cannot touch anything outside its own tables. Data enters it
  only as Arrow batches.

``DuckDBSource.query`` accepts what the SQL guard (``ml.data.sql_guard``) accepts: one
``SELECT`` with no side effects. A user's own DuckDB file is read through
``ml.data.sources.duckdb_file`` instead, which opens it read-only.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import duckdb
import pandas as pd
import pyarrow as pa

# What pandas.read_csv treats as missing, so a file loads with the same nulls as before.
PANDAS_NULL_STRINGS = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]
# No BOOLEAN or DATE here: DuckDB would turn Yes/No columns into booleans, which would
# silently change class labels such as Churn = "Yes". Dates stay text until a task spec
# says which column is the event time.
CSV_TYPE_CANDIDATES = ["BIGINT", "DOUBLE", "VARCHAR"]
BATCH_ROWS = 100_000
DEFAULT_SCHEMA = "main"
INTERNAL_SCHEMA = "mlpilot"  # MLPilot's own derived tables, never the user's data


class EngineError(Exception):
    """Base class for engine failures (raised, never swallowed)."""


class QueryRejected(EngineError):
    """The SQL is not a single SELECT."""


class QueryTimeout(EngineError):
    """The query ran past its time limit and was interrupted."""


class RowLimitExceeded(EngineError):
    """The query returned more rows than the caller allowed."""


@dataclass(frozen=True)
class TableRef:
    name: str
    schema: str = DEFAULT_SCHEMA

    def sql(self) -> str:
        return f"{quote_ident(self.schema)}.{quote_ident(self.name)}"


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True


@dataclass(frozen=True)
class ForeignKey:
    columns: list[str]
    ref_table: TableRef
    ref_columns: list[str]


@dataclass(frozen=True)
class TableSchema:
    table: TableRef
    columns: list[ColumnInfo]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


class DataSource(Protocol):
    """Anything MLPilot can read tables from: a project's DuckDB file now, live databases later."""

    dialect: str  # 'duckdb' | 'postgres' | 'sqlite'

    def list_tables(self) -> list[TableRef]: ...

    def table_schema(self, table: TableRef) -> TableSchema: ...

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table: ...

    def fingerprint(self) -> str:
        """Stable id of the data's current content, for data versions."""
        ...


# Catalog queries for a DuckDB database, shared with read-only user DuckDB files.
DUCKDB_TABLES_SQL = (
    "SELECT table_schema, table_name FROM information_schema.tables "
    "WHERE table_type = 'BASE TABLE' AND table_schema NOT IN (?, 'information_schema') "
    "ORDER BY table_schema, table_name"
)
DUCKDB_COLUMNS_SQL = (
    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
    "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position"
)
DUCKDB_CONSTRAINTS_SQL = (
    "SELECT constraint_type, constraint_column_names, referenced_table, "
    "referenced_column_names FROM duckdb_constraints() "
    "WHERE schema_name = ? AND table_name = ?"
)


def duckdb_table_schema(
    table: TableRef, cols: list[tuple[Any, ...]], constraints: list[tuple[Any, ...]]
) -> TableSchema:
    """A ``TableSchema`` from the rows of ``DUCKDB_COLUMNS_SQL`` and ``DUCKDB_CONSTRAINTS_SQL``."""
    if not cols:
        raise EngineError(f"Table {table.schema}.{table.name} does not exist")
    primary_key: list[str] = []
    foreign_keys: list[ForeignKey] = []
    for kind, columns, ref_table, ref_columns in constraints:
        if kind == "PRIMARY KEY":
            primary_key = list(columns)
        elif kind == "FOREIGN KEY":
            foreign_keys.append(
                ForeignKey(
                    columns=list(columns),
                    ref_table=TableRef(ref_table, table.schema),
                    ref_columns=list(ref_columns),
                )
            )
    return TableSchema(
        table=table,
        columns=[ColumnInfo(n, t, nullable == "YES") for n, t, nullable in cols],
        primary_key=primary_key,
        foreign_keys=foreign_keys,
    )


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_UNSAFE = re.compile(r"[^0-9a-zA-Z_]+")


def table_name_for(filename: str) -> str:
    """A readable SQL table name from a file name: ``Telecom Churn.csv`` -> ``telecom_churn``."""
    stem = _UNSAFE.sub("_", Path(filename).stem).strip("_").lower()
    if not stem:
        stem = "data"
    return f"t_{stem}" if stem[0].isdigit() else stem


def read_file(path: str | Path, *, sep: str = ",") -> pa.RecordBatchReader:
    """Open a CSV or Parquet file as a stream of Arrow batches, typed like pandas would."""
    path = Path(path)
    suffix = path.suffix.lower()
    con = duckdb.connect()  # in memory; the only connection allowed to read files
    if suffix == ".parquet":
        result = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
    elif suffix in (".csv", ".tsv", ".txt"):
        result = con.execute(
            "SELECT * FROM read_csv(?, header=true, delim=?, sample_size=-1, "
            "auto_type_candidates=?, nullstr=?)",
            [str(path), sep, CSV_TYPE_CANDIDATES, PANDAS_NULL_STRINGS],
        )
    else:
        raise EngineError(f"Unsupported file type {suffix!r}; use CSV or Parquet")
    return result.to_arrow_reader(BATCH_ROWS)


def arrow_to_pandas(table: pa.Table) -> pd.DataFrame:
    """The model boundary: Arrow -> pandas, with integer NULLs as float NaN like read_csv."""
    df = table.to_pandas()
    for col in df.columns:
        if str(df[col].dtype) in ("Int64", "Int32", "Int16", "Int8"):
            df[col] = df[col].astype("float64")
    return df


class DuckDBSource:
    """A DuckDB database file (or an in-memory one) read through ``DataSource``."""

    dialect = "duckdb"

    def __init__(self, path: str | Path | None = None, *, read_only: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None and not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(
            str(self.path) if self.path is not None else ":memory:",
            read_only=read_only,
            config={
                "enable_external_access": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            },
        )
        self._lock = threading.Lock()  # one writer at a time; readers use their own cursor

    def close(self) -> None:
        self._con.close()

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = self._fetch(DUCKDB_TABLES_SQL, [INTERNAL_SCHEMA])
        return [TableRef(name=name, schema=schema) for schema, name in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        cols = self._fetch(DUCKDB_COLUMNS_SQL, [table.schema, table.name])
        constraints = self._fetch(DUCKDB_CONSTRAINTS_SQL, [table.schema, table.name])
        return duckdb_table_schema(table, cols, constraints)

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        """Run one SELECT. Raises if it is anything else, runs too long or returns > ``limit`` rows."""
        from ml.data.sql_guard import guard  # the one SQL guard (it imports this module)

        sql = guard(sql, self.dialect).sql
        cursor = self._con.cursor()
        timer = threading.Timer(timeout_s, cursor.interrupt)
        timer.start()
        try:
            reader = cursor.execute(sql, params or []).to_arrow_reader(BATCH_ROWS)
            batches: list[pa.RecordBatch] = []
            rows = 0
            for batch in reader:
                rows += batch.num_rows
                if rows > limit:
                    raise RowLimitExceeded(f"The query returned more than {limit} rows")
                batches.append(batch)
            return pa.Table.from_batches(batches, schema=reader.schema)
        except duckdb.InterruptException as e:
            raise QueryTimeout(f"The query ran longer than {timeout_s} s and was stopped") from e
        finally:
            timer.cancel()
            cursor.close()

    def fingerprint(self) -> str:
        """SHA-256 over every table's name, column types and per-column content hashes."""
        import hashlib

        digest = hashlib.sha256()
        for table in self.list_tables():
            schema = self.table_schema(table)
            digest.update(f"{table.schema}.{table.name}".encode())
            for col in schema.columns:
                digest.update(f"|{col.name}:{col.type}".encode())
            # XOR of per-row hashes: independent of row order, changes if any cell changes.
            row_hash, n_rows = self._fetch(
                f"SELECT coalesce(bit_xor(hash(t)), 0), count(*) FROM {table.sql()} AS t"
            )[0]
            digest.update(f"|{n_rows}:{row_hash}".encode())
        return digest.hexdigest()

    # -- MLPilot's own writes (never built from user text) ---------------------------

    def register_file(self, path: str | Path, table_name: str, *, sep: str = ",") -> TableRef:
        """Load a CSV/Parquet file as a table, replacing a table of the same name."""
        return self.register_arrow(read_file(path, sep=sep), table_name)

    def register_arrow(
        self,
        data: pa.RecordBatchReader | pa.Table,
        table_name: str,
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> TableRef:
        ref = TableRef(name=table_name, schema=schema)
        with self._lock:
            cursor = self._con.cursor()
            try:
                if schema != DEFAULT_SCHEMA:
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")
                cursor.register("incoming", data)
                cursor.execute(f"CREATE OR REPLACE TABLE {ref.sql()} AS SELECT * FROM incoming")
                cursor.unregister("incoming")
            finally:
                cursor.close()
        return ref

    def has_table(self, table: TableRef) -> bool:
        return bool(
            self._fetch(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                [table.schema, table.name],
            )
        )

    def execute(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a statement MLPilot built itself (never user text); returns its rows, if any."""
        with self._lock:
            cursor = self._con.cursor()
            try:
                result = cursor.execute(sql, params or [])
                return result.fetchall() if result.description else []
            finally:
                cursor.close()

    def read_table(self, table: TableRef) -> pa.Table:
        """The whole table as Arrow (MLPilot reading a table it registered itself)."""
        return self._fetch_arrow(f"SELECT * FROM {table.sql()}")

    def row_count(self, table: TableRef) -> int:
        return int(self._fetch(f"SELECT count(*) FROM {table.sql()}")[0][0])

    # -- internals ------------------------------------------------------------------

    def _fetch(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        cursor = self._con.cursor()
        try:
            return cursor.execute(sql, params or []).fetchall()
        finally:
            cursor.close()

    def _fetch_arrow(self, sql: str) -> pa.Table:
        cursor = self._con.cursor()
        try:
            return cursor.execute(sql).to_arrow_table()
        finally:
            cursor.close()
