"""Saved database connections. The password is write-only: no response ever carries it."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from ml.data.profiling.db_stats import TableStats

Dialect = Literal["postgres", "sqlite", "duckdb"]
SslMode = Literal["disable", "prefer", "require", "verify-ca", "verify-full"]


class ConnectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dialect: Dialect
    host: str | None = Field(None, max_length=255, description="Postgres only")
    port: int | None = Field(None, ge=1, le=65535, description="Postgres only (default 5432)")
    database: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Postgres: the database name. SQLite and DuckDB: the path of the file",
    )
    username: str | None = Field(None, max_length=255, description="Postgres only")
    password: SecretStr | None = Field(
        None, description="Stored encrypted; needs MLPILOT_SECRET_KEY. Never returned"
    )
    password_env: str | None = Field(
        None,
        description="Name of an environment variable that holds the password "
        "(nothing secret is stored)",
    )
    ssl_mode: SslMode | None = Field(None, description="Postgres only; default prefer")
    ssl_root_cert: str | None = Field(None, max_length=1024, description="Postgres only")


class ConnectionUpdate(BaseModel):
    """Every field optional; a password given here replaces the saved secret."""

    name: str | None = Field(None, min_length=1, max_length=255)
    host: str | None = Field(None, max_length=255)
    port: int | None = Field(None, ge=1, le=65535)
    database: str | None = Field(None, min_length=1, max_length=255)
    username: str | None = Field(None, max_length=255)
    password: SecretStr | None = None
    password_env: str | None = None
    ssl_mode: SslMode | None = None
    ssl_root_cert: str | None = Field(None, max_length=1024)


class ConnectionRead(BaseModel):
    id: str
    project_id: str
    name: str
    dialect: Dialect
    host: str | None
    port: int | None
    database: str
    username: str | None
    secret: str | None = Field(
        None,
        description="Where the password lives: 'env:NAME' or 'stored (encrypted)'; never its value",
    )
    ssl_mode: SslMode | None
    ssl_root_cert: str | None
    last_tested_at: datetime | None
    can_write: bool | None = Field(
        None, description="From the last test: could the role write? None until tested"
    )
    created_at: datetime


class ConnectionTestResult(BaseModel):
    ok: bool
    error_code: str | None = Field(
        None, description="auth_failed, host_unreachable, timeout, ssl_required, ..."
    )
    message: str | None = Field(None, description="Short and free of secrets")
    server_version: str | None = None
    latency_ms: int | None = None
    can_write: bool | None = Field(None, description="True when the role could modify data")
    privilege_notes: list[str] = []


class TableStatsRead(BaseModel):
    stats: TableStats
    computed_at: datetime
    cached: bool = Field(description="True when these statistics were computed by an earlier call")


class MutableCheckRequest(BaseModel):
    older_version_id: str = Field(description="A snapshot of this database taken earlier")
    newer_version_id: str = Field(description="A snapshot of the same database taken later")
    save: bool = Field(
        True, description="Save the columns found as mutable in the schema overrides"
    )


class MutableColumnRead(BaseModel):
    table: str
    column: str
    compared_rows: int = Field(description="Rows that are in both snapshots")
    changed_rows: int = Field(description="Of those, rows where the column has another value")


class MutableCheckRead(BaseModel):
    mutable: list[MutableColumnRead]
    checked_tables: int
    skipped: list[str] = Field(description="Tables that could not be compared, with the reason")
    saved: bool = Field(description="The columns were saved as mutable in the schema overrides")
    note: str
