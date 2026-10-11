"""MySQL through PyMySQL (MIT, pure Python). Queries go through ``ml.data.sql_guard``, which
runs them in a read-only transaction with ``max_execution_time`` and a row limit.

Same contract as ``postgres.py``. "Schema" in a ``TableRef`` is the MySQL database name.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
from typing import Any

import pyarrow as pa
import pymysql
import pymysql.connections
import pymysql.err

from ml.data import sql_guard
from ml.data.engine import (
    ColumnInfo,
    EngineError,
    ForeignKey,
    QueryTimeout,
    TableRef,
    TableSchema,
)
from ml.data.sources.base import ConnectionFailure, ConnectionSpec, Privileges

DEFAULT_PORT = 3306
CONNECT_TIMEOUT_S = 10
ER_QUERY_TIMEOUT = 3024
# Global, database- or table-level rights that let a user change data, objects or the server.
WRITE_PRIVILEGES = frozenset(
    {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "SUPER", "FILE", "REFERENCES"}
)


def _error_code(exc: BaseException) -> int | None:
    if isinstance(exc, pymysql.err.MySQLError) and exc.args and isinstance(exc.args[0], int):
        return exc.args[0]
    return None


def _failure(exc: BaseException) -> ConnectionFailure:
    """Map a driver error to one of the fixed messages. The driver's own text is dropped."""
    code = _error_code(exc)
    if code in (1045, 1044):
        return ConnectionFailure("auth_failed")
    if code == 1049:
        return ConnectionFailure("database_not_found")
    if code in (ER_QUERY_TIMEOUT, 2013):  # 2013: lost connection (the socket read timed out)
        return ConnectionFailure("timeout")
    if code in (2002, 2003, 2005, 2006):
        return ConnectionFailure("host_unreachable")
    if code is not None:
        return ConnectionFailure("other", f"error code {code}")
    if isinstance(exc, ssl.SSLError):
        return ConnectionFailure("ssl_failed")
    if isinstance(exc, socket.timeout | TimeoutError):
        return ConnectionFailure("timeout")
    if isinstance(exc, OSError):
        return ConnectionFailure("host_unreachable")
    return ConnectionFailure("other")


class MySQLSource:
    """A MySQL database, read-only. A connection is opened per call and always closed."""

    dialect = "mysql"

    def __init__(self, spec: ConnectionSpec) -> None:
        self.spec = spec

    def _ssl_context(self) -> ssl.SSLContext | None:
        mode = self.spec.ssl_mode
        if mode == "disable":
            return None
        ctx = ssl.create_default_context(cafile=self.spec.ssl_root_cert)
        if mode in ("prefer", "require"):
            ctx.check_hostname = False  # encrypted, but the server's identity is not checked
            ctx.verify_mode = ssl.CERT_NONE
        elif mode == "verify-ca":
            ctx.check_hostname = False
        return ctx

    def connect(self, timeout_s: float) -> pymysql.connections.Connection:
        """A new connection (the guard makes the session read-only)."""
        timeout_s = max(timeout_s, 1) + 5  # the socket outlives max_execution_time
        ctx = self._ssl_context()
        try:
            con = pymysql.connect(
                user=self.spec.username or "",
                password=self.spec.password or "",
                host=self.spec.host or "localhost",
                port=self.spec.port or DEFAULT_PORT,
                database=self.spec.database,
                ssl=ctx,
                ssl_disabled=ctx is None,
                connect_timeout=CONNECT_TIMEOUT_S,
                read_timeout=timeout_s,
                write_timeout=timeout_s,
                autocommit=False,
            )
        except (pymysql.err.MySQLError, OSError) as e:
            raise _failure(e) from None
        # PyMySQL carries on in plain text when the server has no SSL; "prefer" accepts that.
        if self.spec.ssl_mode not in ("disable", "prefer") and not getattr(con, "_secure", False):
            con.close()
            raise ConnectionFailure("ssl_failed")
        return con

    def failure(self, exc: BaseException) -> Exception:
        """A driver error as a short fixed message; a statement timeout as ``QueryTimeout``."""
        if isinstance(exc, ConnectionFailure | EngineError):
            return exc
        if _error_code(exc) == ER_QUERY_TIMEOUT:
            return QueryTimeout("The query ran past its time limit and was stopped")
        return _failure(exc)

    # -- DataSource -----------------------------------------------------------------

    def list_tables(self) -> list[TableRef]:
        rows = sql_guard.catalog(
            self,
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type IN ('BASE TABLE', 'VIEW') AND table_schema = DATABASE() "
            "ORDER BY table_name",
        )
        return [TableRef(name=name, schema=schema) for schema, name in rows]

    def table_schema(self, table: TableRef) -> TableSchema:
        params = {"s": table.schema, "t": table.name}
        cols = sql_guard.catalog(
            self,
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position",
            params,
        )
        if not cols:
            raise EngineError(f"Table {table.schema}.{table.name} does not exist")
        keys = sql_guard.catalog(
            self,
            "SELECT constraint_name, column_name, referenced_table_schema, "
            "referenced_table_name, referenced_column_name "
            "FROM information_schema.key_column_usage "
            "WHERE table_schema = :s AND table_name = :t "
            "ORDER BY constraint_name, ordinal_position",
            params,
        )
        pk: list[str] = []
        by_constraint: dict[str, tuple[TableRef, list[str], list[str]]] = {}
        for name, col, ref_schema, ref_table, ref_col in keys:
            if name == "PRIMARY":
                pk.append(col)
            elif ref_table is not None:
                _, cols_, ref_cols = by_constraint.setdefault(
                    name, (TableRef(ref_table, ref_schema), [], [])
                )
                cols_.append(col)
                ref_cols.append(ref_col)
        return TableSchema(
            table=table,
            columns=[ColumnInfo(n, t, nullable == "YES") for n, t, nullable in cols],
            primary_key=pk,
            foreign_keys=[ForeignKey(c, ref, rc) for ref, c, rc in by_constraint.values()],
        )

    def query(
        self, sql: str, params: list[Any] | None = None, *, limit: int, timeout_s: int
    ) -> pa.Table:
        """One SELECT, guarded and read-only. Raises on anything else, a timeout or > limit rows."""
        q = sql_guard.guard(sql, self.dialect)
        return sql_guard.execute(self, q, params=params, limit=limit, timeout_s=timeout_s)

    def fingerprint(self) -> str:
        """SHA-256 of the table and column names and types (schema only, as for Postgres)."""
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
        return str(sql_guard.catalog(self, "SELECT VERSION()")[0][0])

    def privileges(self) -> Privileges:
        """Could this user write? Looks at the global, database and table grants it holds
        directly (privileges that come only through a role are not seen)."""
        me = "grantee = CONCAT('''', REPLACE(CURRENT_USER(), '@', '''@'''), '''')"
        rows = sql_guard.catalog(
            self,
            "SELECT 'server' AS scope, privilege_type FROM information_schema.user_privileges "
            f"WHERE {me} UNION ALL "
            "SELECT CONCAT('database ', table_schema), privilege_type "
            f"FROM information_schema.schema_privileges WHERE {me} UNION ALL "
            "SELECT CONCAT('table ', table_schema, '.', table_name), privilege_type "
            f"FROM information_schema.table_privileges WHERE {me}",
        )
        found: dict[str, list[str]] = {}
        for scope, privilege in rows:
            if str(privilege).upper() in WRITE_PRIVILEGES:
                found.setdefault(str(scope), []).append(str(privilege).upper())
        notes = [f"the user holds {', '.join(sorted(p))} on {s}" for s, p in sorted(found.items())]
        return Privileges(can_write=bool(notes), notes=notes)
