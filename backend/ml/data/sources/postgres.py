"""PostgreSQL through pg8000 (BSD-3, pure Python). Queries go through ``ml.data.sql_guard``,
which runs them in a read-only transaction with a timeout and a row limit.

psycopg 3 is faster but LGPL-3.0, which AGENTS.md rule 9 does not allow as a default;
see docs/adr/0011-connection-secrets-and-drivers.md.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
from typing import Any

import pg8000.exceptions
import pg8000.native
import pyarrow as pa

from ml.data import sql_guard
from ml.data.engine import ColumnInfo, EngineError, QueryTimeout, TableRef, TableSchema
from ml.data.sources.base import ConnectionFailure, ConnectionSpec, Privileges

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

    def connect(self, timeout_s: float) -> pg8000.native.Connection:
        """A new connection (the guard starts the read-only transaction on it)."""
        timeout_s = max(timeout_s, 1) + 5  # the socket outlives statement_timeout
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

    def failure(self, exc: BaseException) -> Exception:
        """A driver error as a short fixed message; a statement timeout as ``QueryTimeout``."""
        if isinstance(exc, ConnectionFailure | EngineError):
            return exc
        if _error_code(exc) == "57014":
            return QueryTimeout("The query ran past its time limit and was stopped")
        return _failure(exc)

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = sql_guard.catalog(
            self,
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type IN ('BASE TABLE', 'VIEW') "
            f"AND table_schema NOT IN {SYSTEM_SCHEMAS!r} AND table_schema NOT LIKE 'pg_%' "
            "ORDER BY table_schema, table_name",
        )
        return [TableRef(name=name, schema=schema) for schema, name in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        cols = sql_guard.catalog(
            self,
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position",
            {"s": table.schema, "t": table.name},
        )
        if not cols:
            raise EngineError(f"Table {table.schema}.{table.name} does not exist")
        pk = sql_guard.catalog(
            self,
            "SELECT kcu.column_name FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = :s "
            "AND tc.table_name = :t ORDER BY kcu.ordinal_position",
            {"s": table.schema, "t": table.name},
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
        """One SELECT, guarded and read-only. Raises on anything else, a timeout or > limit rows."""
        q = sql_guard.guard(sql, self.dialect)
        return sql_guard.execute(self, q, params=params, limit=limit, timeout_s=timeout_s)

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
        sql_guard.catalog(self, "SELECT 1")

    def server_version(self) -> str:
        return str(sql_guard.catalog(self, "SELECT current_setting('server_version')")[0][0])

    def privileges(self) -> Privileges:
        """Could this role write? Looks at superuser, create rights and write grants."""
        notes: list[str] = []
        ((is_super, can_createdb, can_createrole, db_create, schemas, writable),) = (
            sql_guard.catalog(
                self,
                "SELECT r.rolsuper, r.rolcreatedb, r.rolcreaterole, "
                "has_database_privilege(current_user, current_database(), 'CREATE'), "
                "(SELECT string_agg(n.nspname, ', ' ORDER BY n.nspname) FROM pg_namespace AS n "
                "WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' "
                "AND has_schema_privilege(current_user, n.oid, 'CREATE')), "
                "(SELECT count(*) FROM pg_class AS c JOIN pg_namespace AS n "
                "ON n.oid = c.relnamespace WHERE c.relkind IN ('r', 'p') "
                "AND n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' AND ("
                "has_table_privilege(current_user, c.oid, 'INSERT') OR "
                "has_table_privilege(current_user, c.oid, 'UPDATE') OR "
                "has_table_privilege(current_user, c.oid, 'DELETE') OR "
                "has_table_privilege(current_user, c.oid, 'TRUNCATE'))) "
                "FROM pg_roles AS r WHERE r.rolname = current_user",
            )
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
            notes.append(f"the role can create objects in schema(s): {schemas}")
        if writable:
            notes.append(f"the role can write to {writable} table(s)")
        return Privileges(can_write=bool(notes), notes=notes)
