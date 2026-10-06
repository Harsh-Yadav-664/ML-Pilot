"""Dataset versions as the API sees them: resolve, profile, scan for leakage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import DataVersion
from app.schemas.api import ColumnProfile, DataMetrics, LeakageFinding
from ml.data.profiling.profiler import DataProfiler
from ml.data.versions import content_hash
from ml.data.workspace import workspace_for
from ml.validation.leakage import scan

# The UI's severity scale: block -> high, warn -> medium, info -> low.
# A version id is the SHA-256 of the file, so it can never name a path outside the store.
VERSION_ID = re.compile(r"[0-9a-f]{64}")

UI_SEVERITY = {"block": "high", "warn": "medium", "info": "low"}


async def version_path(db: AsyncSession, data_version_id: str) -> Path:
    """The stored file of a data version; HTTP 404 if there is none.

    Versions are content-addressed, so identical bytes loaded in two projects are one
    version (recorded under the project that loaded it first). Any project may use it.
    """
    if not VERSION_ID.fullmatch(data_version_id):
        raise HTTPException(status_code=404, detail="Data version not found")
    path = datasets.VERSIONS_DIR / f"{data_version_id}.csv"
    if await db.get(DataVersion, data_version_id) is None or not path.exists():
        raise HTTPException(status_code=404, detail="Data version not found")
    return path


def load(path: Path, project_id: str, filename: str | None = None) -> pd.DataFrame:
    """The file as a DataFrame, read from the project's DuckDB table for it."""
    try:
        return workspace_for(project_id, datasets.PROJECTS_DIR).load(path, filename)
    except Exception as e:  # any read error becomes an HTTP 400
        raise HTTPException(status_code=400, detail=f"Could not load the dataset: {e}") from e


@dataclass(frozen=True)
class Registered:
    version_id: str
    table: str
    columns: list[str]
    rows: int


def register(path: Path, project_id: str, filename: str) -> Registered:
    """Load a file into the project's DuckDB as a table; HTTP 400 if it can't be parsed."""
    ws = workspace_for(project_id, datasets.PROJECTS_DIR)
    try:
        version_id = content_hash(path)
        table = ws.register_version(path, version_id, filename)
        return Registered(
            version_id=version_id,
            table=table.name,
            columns=[c.name for c in ws.source.table_schema(table).columns],
            rows=ws.source.row_count(table),
        )
    except Exception as e:  # any read error becomes an HTTP 400
        raise HTTPException(status_code=400, detail=f"Could not load the dataset: {e}") from e


def metrics(df: pd.DataFrame, target_column: str) -> DataMetrics:
    profile = DataProfiler().profile(df, target_column=target_column)
    return DataMetrics(
        total_rows=profile.rows,
        total_columns=profile.columns,
        missing_data_percent=float(profile.missing_rate),
        duplicate_rows=profile.duplicate_rows,
    )


def column_profile(series: pd.Series, name: str, target_column: str, n_rows: int) -> ColumnProfile:
    """Real per-column summary for the UI: dtype, missing %, unique count, 12-bin shape."""
    non_null = series.dropna()
    unique = int(non_null.nunique())
    if pd.api.types.is_bool_dtype(series):
        dtype = "bool"
    elif pd.api.types.is_integer_dtype(series):
        dtype = "int"
    elif pd.api.types.is_float_dtype(series):
        dtype = "float"
    elif pd.api.types.is_datetime64_any_dtype(series):
        dtype = "datetime"
    elif unique <= 50:
        dtype = "category"
    else:
        dtype = "string"

    if dtype in ("int", "float") and unique > 12:
        counts, _ = np.histogram(non_null.astype(float), bins=12)
        dist = counts.tolist()
    else:
        dist = non_null.astype(str).value_counts().head(12).tolist()
    peak = max(dist) if dist else 0
    dist = [round(c / peak, 4) if peak else 0.0 for c in dist] + [0.0] * (12 - len(dist))

    if name == target_column:
        role = "target"
    elif dtype in ("string", "int") and n_rows > 0 and unique == n_rows:
        role = "id"
    else:
        role = "feature"
    return ColumnProfile(
        name=name,
        dtype=dtype,
        role=role,
        missing_pct=round(float(series.isna().mean()) * 100, 2),
        unique=unique,
        dist=dist,
    )


def columns(df: pd.DataFrame, target_column: str) -> list[ColumnProfile]:
    return [column_profile(df[c], c, target_column, len(df)) for c in df.columns]


def leakage(df: pd.DataFrame, target_column: str) -> list[LeakageFinding]:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column {target_column!r} not found")
    return [
        LeakageFinding(
            id=f"warn_{i}",
            column=f.column or "(rows)",
            message=f.explanation,
            severity=UI_SEVERITY[f.severity],  # type: ignore[arg-type]
            category=f.category,
            check=f.check,
            evidence=f.evidence,
        )
        for i, f in enumerate(scan(df, target_column))
    ]
