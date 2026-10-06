"""Register immutable dataset versions (ml/data/versions.py) in the metadata DB."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import DataVersion
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
