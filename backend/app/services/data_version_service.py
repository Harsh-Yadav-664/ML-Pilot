"""Register immutable dataset versions (ml/data/versions.py) in the metadata DB."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import DataVersion
from app.schemas.snapshot import DbVersionRead
from ml.data.snapshot import DataDescription, DataVersionInfo
from ml.data.versions import StoredVersion, store_version


async def register_file_version(
    db: AsyncSession,
    source: Path,
    project_id: str,
    kind: str,
    origin: dict[str, Any],
    n_rows: int,
    n_columns: int,
) -> StoredVersion:
    """Store ``source`` as a read-only versioned copy and record it (once per content).

    ``origin`` says where the data came from (upload name, sample name, connection
    name and query). It must never contain secrets.
    """
    stored = await asyncio.to_thread(store_version, source, datasets.VERSIONS_DIR)
    if await db.get(DataVersion, stored.id) is None:
        db.add(
            DataVersion(
                id=stored.id,
                project_id=project_id,
                kind=kind,
                source=origin,
                n_rows=n_rows,
                n_columns=n_columns,
            )
        )
        await db.flush()
    return stored


def version_fields(stored: StoredVersion) -> dict[str, Any]:
    """What API responses expose about a version."""
    return {
        "dataset_path": str(stored.path),
        "data_version_id": stored.id,
        "short_hash": stored.short_hash,
    }


DB_KINDS = {"snapshot": "db_snapshot", "live": "db_live"}


async def record_database_version(
    db: AsyncSession, project_id: str, info: DataVersionInfo
) -> DataVersion:
    """Record a snapshot or live version of a database (once per id).

    ``source`` is the description itself: mode, connection id and name, ``as_of``, and per table
    the columns, row count, latest event time and checksum. It holds no secrets.
    """
    row = await db.get(DataVersion, info.id)
    if row is None:
        description = info.description
        row = DataVersion(
            id=info.id,
            project_id=project_id,
            kind=DB_KINDS[description.mode],
            source=description.model_dump(mode="json"),
            n_rows=description.n_rows,
            n_columns=description.n_columns,
            max_event_time=description.max_event_times,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
    return row


def database_version_read(row: DataVersion) -> DbVersionRead:
    description = DataDescription.model_validate(row.source)
    return DbVersionRead(
        id=row.id,
        short_hash=row.id[:12],
        kind=row.kind,  # type: ignore[arg-type]
        mode=description.mode,
        as_of=description.as_of,
        connection=description.connection,
        reproducible=description.reproducible,
        note=description.note,
        n_rows=description.n_rows,
        n_columns=description.n_columns,
        tables=description.tables,
        created_at=row.created_at,
    )


async def database_description(
    db: AsyncSession, data_version_id: str | None
) -> DataDescription | None:
    """What a run manifest says about its data when that is a database version, else None."""
    if not data_version_id:
        return None
    row = await db.get(DataVersion, data_version_id)
    if row is None or row.kind not in DB_KINDS.values():
        return None
    return DataDescription.model_validate(row.source)
