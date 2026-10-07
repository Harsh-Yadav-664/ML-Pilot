"""Request and response models for data versions of a database (#97)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.api import JobRead
from ml.data.snapshot import TableRecord


class SnapshotRequest(BaseModel):
    mode: Literal["snapshot", "live"] = Field(
        "snapshot",
        description="snapshot copies the tables into the project; live copies nothing and only "
        "records the state of the source",
    )
    tables: list[str] | None = Field(
        None, description="Table keys from the schema graph; default all"
    )
    columns: dict[str, list[str]] | None = Field(
        None, description="Columns per table key; default all. The time column is always read."
    )
    as_of: datetime | None = Field(
        None, description="Upper bound on every event time (UTC); default now"
    )


class DbVersionRead(BaseModel):
    id: str
    short_hash: str
    kind: Literal["db_snapshot", "db_live"]
    mode: Literal["snapshot", "live"]
    as_of: datetime
    connection: dict[str, str]
    reproducible: bool
    note: str | None
    n_rows: int
    n_columns: int
    tables: dict[str, TableRecord]
    created_at: datetime


class SnapshotResponse(BaseModel):
    """A live version is ready at once; a snapshot is a job (poll `GET /jobs/{id}`)."""

    version: DbVersionRead | None = None
    job: JobRead | None = None
