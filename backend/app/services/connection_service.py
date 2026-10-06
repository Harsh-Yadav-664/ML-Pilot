"""Saved connections: create, read, test and open them without ever exposing the password."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets, secrets
from app.db.models.connection import Connection
from app.schemas.connection import (
    ConnectionCreate,
    ConnectionRead,
    ConnectionTestResult,
    ConnectionUpdate,
)
from ml.data.engine import EngineError
from ml.data.schema_graph import (
    OverrideError,
    SchemaGraph,
    SchemaOverrides,
    apply_overrides,
    build_schema_graph,
    merge_overrides,
    validate_overrides,
)
from ml.data.sources import ConnectionFailure, ConnectionSpec, SourceWithChecks, open_source
from ml.data.sources.duckdb_file import is_duckdb_file
from ml.data.sources.sqlite import is_sqlite_file

logger = logging.getLogger(__name__)

FILE_CHECKS = {"sqlite": is_sqlite_file, "duckdb": is_duckdb_file}


def check_database_file(dialect: str, path: str) -> str:
    """The resolved path of a SQLite/DuckDB file, or HTTP 400 if it isn't a usable one.

    The file must exist and carry the right header, so this can't be used to open some other
    file, and it can't be one of MLPilot's own (a project's work.duckdb or a stored version).
    """
    resolved = Path(path).expanduser().resolve()
    for own in (datasets.PROJECTS_DIR, datasets.VERSIONS_DIR):
        if resolved.is_relative_to(Path(own).resolve()):
            raise HTTPException(400, "That file belongs to MLPilot itself and can't be connected")
    if not resolved.is_file() or not FILE_CHECKS[dialect](resolved):
        raise HTTPException(400, f"{path!r} is not a readable {dialect} database file")
    return str(resolved)


def to_read(conn: Connection) -> ConnectionRead:
    ssl = conn.ssl or {}
    return ConnectionRead(
        id=conn.id,
        project_id=conn.project_id,
        name=conn.name,
        dialect=conn.dialect,  # type: ignore[arg-type]
        host=conn.host,
        port=conn.port,
        database=conn.database or "",
        username=conn.username,
        secret=secrets.describe(conn.secret_ref),
        ssl_mode=ssl.get("mode"),
        ssl_root_cert=ssl.get("root_cert"),
        last_tested_at=conn.last_tested_at,
        can_write=conn.can_write,
        created_at=conn.created_at,
    )


def _secret_ref(password: str | None, password_env: str | None) -> str | None:
    if password is not None and password_env is not None:
        raise HTTPException(400, "Give either password or password_env, not both")
    try:
        if password is not None:
            return secrets.store_secret(password)
        if password_env is not None:
            return secrets.env_ref(password_env)
    except secrets.SecretError as e:
        raise HTTPException(400, str(e)) from None
    return None


class ConnectionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, project_id: str, connection_id: str) -> Connection:
        conn = await self.db.get(Connection, connection_id)
        if conn is None or conn.project_id != project_id:
            raise HTTPException(404, "Connection not found")
        return conn

    async def list(self, project_id: str) -> list[Connection]:
        result = await self.db.execute(
            select(Connection)
            .where(Connection.project_id == project_id)
            .order_by(Connection.created_at, Connection.name)
        )
        return list(result.scalars())

    async def _require_unique_name(
        self, project_id: str, name: str, *, keep: str | None = None
    ) -> None:
        for other in await self.list(project_id):
            if other.name == name and other.id != keep:
                raise HTTPException(409, f"A connection named {name!r} already exists")

    async def create(self, project_id: str, data: ConnectionCreate) -> Connection:
        await self._require_unique_name(project_id, data.name)
        database = data.database
        if data.dialect == "postgres":
            if not (data.host and data.username):
                raise HTTPException(400, "A Postgres connection needs host and username")
        else:
            database = check_database_file(data.dialect, database)
        ref = _secret_ref(
            data.password.get_secret_value() if data.password else None, data.password_env
        )
        conn = Connection(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=data.name,
            dialect=data.dialect,
            host=data.host,
            port=data.port,
            database=database,
            username=data.username,
            secret_ref=ref,
            ssl=_ssl_json(data.ssl_mode, data.ssl_root_cert)
            if data.dialect == "postgres"
            else None,
        )
        self.db.add(conn)
        await self.db.flush()
        await self.db.refresh(conn)
        logger.info("Saved %s connection %r (%s)", conn.dialect, conn.name, conn.id)
        return conn

    async def update(
        self, project_id: str, connection_id: str, data: ConnectionUpdate
    ) -> Connection:
        conn = await self.get(project_id, connection_id)
        if data.name is not None and data.name != conn.name:
            await self._require_unique_name(project_id, data.name, keep=conn.id)
            conn.name = data.name
        for field in ("host", "port", "username"):
            value = getattr(data, field)
            if value is not None:
                setattr(conn, field, value)
        if data.database is not None:
            conn.database = (
                data.database
                if conn.dialect == "postgres"
                else check_database_file(conn.dialect, data.database)
            )
        if data.password is not None or data.password_env is not None:
            conn.secret_ref = _secret_ref(
                data.password.get_secret_value() if data.password else None, data.password_env
            )
        if data.ssl_mode is not None or data.ssl_root_cert is not None:
            current = conn.ssl or {}
            conn.ssl = _ssl_json(
                data.ssl_mode or current.get("mode"), data.ssl_root_cert or current.get("root_cert")
            )
        conn.last_tested_at = None  # what was tested is no longer what is saved
        conn.can_write = None
        await self.db.flush()
        return conn

    async def delete(self, project_id: str, connection_id: str) -> None:
        conn = await self.get(project_id, connection_id)
        await self.db.delete(conn)
        await self.db.flush()

    # -- using a connection ---------------------------------------------------------

    def spec(self, conn: Connection) -> ConnectionSpec:
        """The connection with its password resolved. HTTP 400 if the secret can't be read."""
        try:
            password = secrets.resolve_secret(conn.secret_ref)
        except secrets.SecretError as e:
            raise HTTPException(400, str(e)) from None
        ssl = conn.ssl or {}
        database = conn.database or ""
        if conn.dialect != "postgres":
            database = check_database_file(conn.dialect, database)
        return ConnectionSpec(
            dialect=conn.dialect,  # type: ignore[arg-type]
            database=database,
            host=conn.host,
            port=conn.port,
            username=conn.username,
            password=password,
            ssl_mode=ssl.get("mode") or "prefer",
            ssl_root_cert=ssl.get("root_cert"),
        )

    def source(self, conn: Connection) -> SourceWithChecks:
        return open_source(self.spec(conn))

    async def schema_graph(self, conn: Connection) -> SchemaGraph:
        """Tables, keys and time columns of the database, with the saved overrides applied."""
        overrides = SchemaOverrides.model_validate(conn.schema_overrides or {})
        return apply_overrides(await self._read_schema(conn), overrides)

    async def update_schema_overrides(
        self, conn: Connection, patch: SchemaOverrides
    ) -> SchemaGraph:
        """Check ``patch`` against the live schema, save it, return the graph it produces."""
        merged = merge_overrides(conn.schema_overrides, patch)
        base = await self._read_schema(conn)
        try:
            validate_overrides(base, merged)
        except OverrideError as e:
            raise HTTPException(422, str(e)) from None
        conn.schema_overrides = merged.model_dump(mode="json")
        await self.db.flush()
        return apply_overrides(base, merged)

    async def _read_schema(self, conn: Connection) -> SchemaGraph:
        source = self.source(conn)
        try:
            return await asyncio.to_thread(build_schema_graph, source)
        except (ConnectionFailure, EngineError) as e:
            raise HTTPException(502, f"Could not read the schema: {e}") from None

    async def test(self, conn: Connection) -> ConnectionTestResult:
        """Connect, read the server version and the role's privileges. Failures are reported
        in the result (``ok=False`` with a short code and message), never raised."""
        source = self.source(conn)

        def run() -> ConnectionTestResult:
            started = time.perf_counter()
            try:
                source.check()
                version = source.server_version()
                latency = round((time.perf_counter() - started) * 1000)
                privileges = source.privileges()
            except ConnectionFailure as e:
                return ConnectionTestResult(ok=False, error_code=e.code, message=e.message)
            return ConnectionTestResult(
                ok=True,
                server_version=version,
                latency_ms=latency,
                can_write=privileges.can_write,
                privilege_notes=privileges.notes,
            )

        result = await asyncio.to_thread(run)
        conn.last_tested_at = datetime.now(UTC)
        conn.can_write = result.can_write
        await self.db.flush()
        logger.info("Tested connection %r (%s): ok=%s", conn.name, conn.id, result.ok)
        return result


def _ssl_json(mode: str | None, root_cert: str | None) -> dict[str, str] | None:
    out = {k: v for k, v in (("mode", mode), ("root_cert", root_cert)) if v}
    return out or None
