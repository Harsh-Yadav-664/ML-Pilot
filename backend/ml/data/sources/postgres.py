"""PostgreSQL through pg8000 (BSD-3, pure Python): read-only transactions, timeout, row limit.

psycopg 3 is faster but LGPL-3.0, which AGENTS.md rule 9 does not allow as a default;
see docs/adr/0011-connection-secrets-and-drivers.md.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pg8000.exceptions
import pg8000.native
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
    Privileges,
    guard,
    limited_sql,
    rows_to_arrow,
)

DEFAULT_PORT = 5432
CONNECT_TIMEOUT_S = 10
SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")


def _error_code(exc: BaseException) -> str | None:
    """The SQLSTATE of a pg8000 server error, if it carries one."""
    if exc.args and isinstance(exc.args[0], dict):
        code = exc.args[0].get("C")
        return str(code) if code else None
    return None


def _failure(exc: BaseException) -> ConnectionFailure:
    """Map a driver error to one of the fixed messages. The driver's own text is dropped."""
    code = _error_code(exc)
    server_text = (
        str(exc.args[0].get("M", "")) if exc.args and isinstance(exc.args[0], dict) else ""
    )
    if code in ("28P01", "28000"):
        # 28000 is also what a pg_hba rule that demands SSL answers with.
        if "ssl" in server_text.lower() or "encryption" in server_text.lower():
            return ConnectionFailure("ssl_required")
        return ConnectionFailure("auth_failed")
    if code == "3D000":
        return ConnectionFailure("database_not_found")
    if code == "57014":
        return ConnectionFailure("timeout")
    if code is not None:
        return ConnectionFailure("other", f"error code {code}")
    if isinstance(exc, ssl.SSLError):
        return ConnectionFailure("ssl_failed")
    if isinstance(exc, socket.timeout | TimeoutError):
        return ConnectionFailure("timeout")
    if isinstance(exc, OSError | pg8000.exceptions.InterfaceError):
        return ConnectionFailure("host_unreachable")
    return ConnectionFailure("other")


class PostgresSource:
    """A PostgreSQL database, read-only. A connection is opened per call and always closed."""

    dialect = "postgres"

    def __init__(self, spec: ConnectionSpec) -> None:
        self.spec = spec

    # -- connection -----------------------------------------------------------------

    def _ssl_context(self, mode: str) -> ssl.SSLContext | None:
        if mode == "disable":
            return None
        ctx = ssl.create_default_context(cafile=self.spec.ssl_root_cert)
        if mode in ("prefer", "require"):
            ctx.check_hostname = False  # encrypted, but the server's identity is not checked
            ctx.verify_mode = ssl.CERT_NONE
        elif mode == "verify-ca":
            ctx.check_hostname = False
        return ctx

    def _open(self, timeout_s: float) -> pg8000.native.Connection:
        modes = ["require", "disable"] if self.spec.ssl_mode == "prefer" else [self.spec.ssl_mode]
        last: BaseException | None = None
        for mode in modes:
            try:
                return pg8000.native.Connection(
                    user=self.spec.username or "",
                    password=self.spec.password,
                    host=self.spec.host or "localhost",
                    port=self.spec.port or DEFAULT_PORT,
                    database=self.spec.database,
                    ssl_context=self._ssl_context(mode),
                    timeout=timeout_s,
                )
            except pg8000.exceptions.InterfaceError as e:
                if "SSL" in str(e):
                    if len(modes) > 1 and mode == "require":
                        last = e  # "prefer": the server has no SSL, so try without
                        continue
                    raise ConnectionFailure("ssl_failed") from None  # the server refuses SSL
                raise _failure(e) from None
            except (pg8000.exceptions.DatabaseError, OSError, ssl.SSLError) as e:
                raise _failure(e) from None
        assert last is not None
        raise _failure(last) from None

    @contextmanager
    def _read_only(self, timeout_s: float) -> Iterator[pg8000.native.Connection]:
        """A connection inside a READ ONLY transaction with a statement timeout; always rolled back."""
        conn = self._open(max(timeout_s, 1) + 5)
        try:
            conn.run("START TRANSACTION READ ONLY")
            conn.run(f"SET LOCAL statement_timeout = {int(timeout_s) * 1000}")
            try:
                yield conn
            finally:
                conn.run("ROLLBACK")
        except (pg8000.exceptions.DatabaseError, OSError) as e:
            raise _failure(e) from None
        finally:
            conn.close()

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = self._rows(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type IN ('BASE TABLE', 'VIEW') "
            f"AND table_schema NOT IN {SYSTEM_SCHEMAS!r} AND table_schema NOT LIKE 'pg_%' "
            "ORDER BY table_schema, table_name"
        )
        return [TableRef(name=name, schema=schema) for schema, name in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        with self._read_only(15) as conn:
            cols = conn.run(
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position",
                s=table.schema,
                t=table.name,
            )
            if not cols:
                raise EngineError(f"Table {table.schema}.{table.name} does not exist")
            pk = conn.run(
                "SELECT kcu.column_name FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
                "WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = :s "
                "AND tc.table_name = :t ORDER BY kcu.ordinal_position",
                s=table.schema,
                t=table.name,
            )
        # Foreign keys come with schema introspection (#45).
        return TableSchema(
            table=table,
            columns=[ColumnInfo(n, t, nullable == "YES") for n, t, nullable in cols],
            primary_key=[r[0] for r in pk],
        )

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        """One SELECT in a read-only transaction. Raises on anything else, a timeout or > limit rows."""
        if params:
            raise EngineError("Query parameters are not supported for Postgres sources")
        guard(sql, "postgres")
        wrapped = limited_sql(sql, "postgres", limit)
        try:
            with self._read_only(timeout_s) as conn:
                rows = conn.run(wrapped)
                names = [c["name"] for c in conn.columns or []]
        except ConnectionFailure as e:
            if e.code == "timeout":
                raise QueryTimeout(e.message) from None
            raise
        if len(rows) > limit:
            raise RowLimitExceeded(f"The query returned more than {limit} rows")
        return rows_to_arrow(names, [tuple(r) for r in rows])

    def fingerprint(self) -> str:
        """SHA-256 of the table and column names and types. Schema only: the data of a live
        database changes under us, so a content version comes from a snapshot (#97)."""
        digest = hashlib.sha256()
        for table in self.list_tables():
            digest.update(f"{table.schema}.{table.name}".encode())
            for col in self.table_schema(table).columns:
                digest.update(f"|{col.name}:{col.type}".encode())
        return digest.hexdigest()

    # -- checks used by the connection manager --------------------------------------

    def check(self) -> None:
        self._rows("SELECT 1")

    def server_version(self) -> str:
        return str(self._rows("SHOW server_version")[0][0])

    def privileges(self) -> Privileges:
        """Could this role write? Looks at superuser, create rights and write grants."""
        notes: list[str] = []
        with self._read_only(15) as conn:
            ((is_super, can_createdb, can_createrole),) = conn.run(
                "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles "
                "WHERE rolname = current_user"
            )
            schemas = [
                r[0]
                for r in conn.run(
                    "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' "
                    "AND nspname <> 'information_schema' "
                    "AND has_schema_privilege(current_user, oid, 'CREATE') ORDER BY 1"
                )
            ]
            ((writable,),) = conn.run(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relkind IN ('r', 'p') AND n.nspname NOT LIKE 'pg_%' "
                "AND n.nspname <> 'information_schema' AND ("
                "has_table_privilege(current_user, c.oid, 'INSERT') OR "
                "has_table_privilege(current_user, c.oid, 'UPDATE') OR "
                "has_table_privilege(current_user, c.oid, 'DELETE') OR "
                "has_table_privilege(current_user, c.oid, 'TRUNCATE'))"
            )
            ((db_create,),) = conn.run(
                "SELECT has_database_privilege(current_user, current_database(), 'CREATE')"
            )
        if is_super:
            notes.append("the role is a superuser")
        if can_createdb:
            notes.append("the role can create databases")
        if can_createrole:
            notes.append("the role can create roles")
        if db_create:
            notes.append("the role can create schemas in this database")
        if schemas:
            notes.append("the role can create objects in schema(s): " + ", ".join(schemas))
        if writable:
            notes.append(f"the role can write to {writable} table(s)")
        return Privileges(can_write=bool(notes), notes=notes)

    # -- internals ------------------------------------------------------------------

    def _rows(self, sql: str) -> list[list[Any]]:
        with self._read_only(15) as conn:
            return [list(r) for r in conn.run(sql)]
