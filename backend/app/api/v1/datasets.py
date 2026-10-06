"""Datasets of a project: every load becomes an immutable data version (#39)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import DBSession, Gateway, ProjectID
from app.core import datasets
from app.schemas.api import (
    ColumnProfile,
    DataMetrics,
    DatasetInfo,
    FeatureSuggestion,
    LeakageFinding,
    SampleDatasetRequest,
    SqlSnapshotRequest,
)
from app.services import data_service
from app.services.data_version_service import register_file_version
from app.services.experiment_service import ExperimentService
from ml.data.ingestion.sql_loader import SqlLoader, redact
from ml.data.versions import StoredVersion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])


def _info(stored: StoredVersion, filename: str, columns: list[str], rows: int) -> DatasetInfo:
    return DatasetInfo(
        data_version_id=stored.id,
        short_hash=stored.short_hash,
        filename=filename,
        columns=columns,
        total_rows=rows,
    )


@router.post("/upload", response_model=DatasetInfo)
async def upload_dataset(
    project_id: ProjectID, db: DBSession, file: UploadFile = File(...)
) -> DatasetInfo:
    """Upload a CSV, store it as an immutable version and return its columns."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    # Keep only the base name so a crafted filename can't escape the upload dir
    safe_name = os.path.basename(file.filename)
    os.makedirs(datasets.UPLOAD_DIR, exist_ok=True)
    file_path = Path(datasets.UPLOAD_DIR) / f"{uuid.uuid4()}_{safe_name}"

    def save_upload() -> None:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    await asyncio.to_thread(save_upload)
    try:
        loaded = await asyncio.to_thread(data_service.register, file_path, project_id, safe_name)
        stored = await register_file_version(
            db,
            file_path,
            project_id,
            kind="file",
            origin={"type": "upload", "filename": safe_name, "table": loaded.table},
            n_rows=loaded.rows,
            n_columns=len(loaded.columns),
        )
    finally:
        # The versioned copy is what experiments use; the upload itself isn't kept.
        file_path.unlink(missing_ok=True)
    return _info(stored, safe_name, loaded.columns, loaded.rows)


@router.post("/sample", response_model=DatasetInfo)
async def load_sample_dataset(
    request: SampleDatasetRequest, project_id: ProjectID, db: DBSession
) -> DatasetInfo:
    """Load a bundled sample dataset as an immutable version."""
    filename = f"{request.dataset_name}.csv"
    source = (Path(datasets.DATASETS_DIR) / filename).resolve()
    if source.parent != Path(datasets.DATASETS_DIR).resolve() or not source.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    loaded = await asyncio.to_thread(data_service.register, source, project_id, filename)
    columns = loaded.columns
    stored = await register_file_version(
        db,
        source,
        project_id,
        kind="file",
        origin={"type": "sample", "name": request.dataset_name, "table": loaded.table},
        n_rows=loaded.rows,
        n_columns=len(columns),
    )
    info = _info(stored, filename, columns, loaded.rows)
    # Pre-select a likely target for the UI to be helpful
    info.default_target = "Churn" if "Churn" in columns else (columns[-1] if columns else None)
    return info


@router.post("/sql", response_model=DatasetInfo)
async def snapshot_sql_query(
    request: SqlSnapshotRequest, project_id: ProjectID, db: DBSession
) -> DatasetInfo:
    """Run one read-only query and store the result as an immutable version."""
    loader = SqlLoader()
    try:
        df = loader.load(request.connection_string, request.query)
    except Exception as e:  # noqa: BLE001 - any driver error becomes a redacted HTTP 400
        # Never echo the connection string or password back or into logs.
        message = redact(str(e), request.connection_string)
        logger.error(f"Failed to connect and query SQL: {message}")
        raise HTTPException(
            status_code=400, detail=f"Failed to connect and query SQL: {message}"
        ) from None
    file_id = loader.hash_connection(request.connection_string, request.query)
    os.makedirs(datasets.UPLOAD_DIR, exist_ok=True)
    file_path = Path(datasets.UPLOAD_DIR) / f"sql_{file_id}.csv"
    df.to_csv(file_path, index=False)
    try:
        loaded = await asyncio.to_thread(
            data_service.register, file_path, project_id, f"sql_{file_id}.csv"
        )
        stored = await register_file_version(
            db,
            file_path,
            project_id,
            kind="db_snapshot",
            # The connection is identified by its hash only: no host, user or password.
            origin={
                "type": "sql",
                "connection": file_id,
                "query": request.query,
                "table": loaded.table,
            },
            n_rows=loaded.rows,
            n_columns=len(loaded.columns),
        )
    finally:
        file_path.unlink(missing_ok=True)
    return _info(stored, f"SQL Query Snapshot ({file_id})", df.columns.tolist(), len(df))


@router.get("/{data_version_id}/metrics", response_model=DataMetrics)
async def get_metrics(
    project_id: ProjectID, data_version_id: str, target_column: str, db: DBSession
) -> DataMetrics:
    df = data_service.load(await data_service.version_path(db, data_version_id), project_id)
    return data_service.metrics(df, target_column)


@router.get("/{data_version_id}/columns", response_model=list[ColumnProfile])
async def get_columns(
    project_id: ProjectID, data_version_id: str, target_column: str, db: DBSession
) -> list[ColumnProfile]:
    df = data_service.load(await data_service.version_path(db, data_version_id), project_id)
    return data_service.columns(df, target_column)


@router.get("/{data_version_id}/leakage", response_model=list[LeakageFinding])
async def get_leakage(
    project_id: ProjectID, data_version_id: str, target_column: str, db: DBSession
) -> list[LeakageFinding]:
    df = data_service.load(await data_service.version_path(db, data_version_id), project_id)
    return await asyncio.to_thread(data_service.leakage, df, target_column)


@router.get("/{data_version_id}/suggestions", response_model=list[FeatureSuggestion])
async def get_suggestions(
    project_id: ProjectID,
    data_version_id: str,
    target_column: str,
    db: DBSession,
    gateway: Gateway,
    user_query: str | None = None,
) -> list[FeatureSuggestion]:
    """Ask the planner for feature ideas (data only; nothing is run)."""
    path = await data_service.version_path(db, data_version_id)
    objective = "Maximize F1 score while preventing overfitting"
    if user_query:
        objective += (
            f". User strictly requested: '{user_query}'. Generate features reflecting this."
        )
    try:
        hypotheses = await ExperimentService(db).suggest_experiments(
            dataset_version=str(path),
            project_id=project_id,
            target_column=target_column,
            objective=objective,
            max_hypotheses=3,
            gateway=gateway,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=502, detail=f"AI suggestions failed: {e}") from e
    return [FeatureSuggestion.model_validate(h) for h in hypotheses]
