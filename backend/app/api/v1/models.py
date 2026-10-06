"""Model artifacts API endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import DBSession
from app.db.models.model_artifact import ModelArtifact

router = APIRouter(prefix="/model-artifacts", tags=["models"])


class ModelArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    experiment_id: str
    project_id: str
    model_name: str
    file_path: str
    format: str
    metrics: dict | None = None
    feature_importance: dict | None = None
    size_bytes: float | None = None
    created_at: datetime


@router.get("/experiment/{experiment_id}", response_model=list[ModelArtifactRead])
async def list_model_artifacts(experiment_id: str, db: DBSession) -> list[ModelArtifactRead]:
    result = await db.execute(
        select(ModelArtifact).where(ModelArtifact.experiment_id == experiment_id)
    )
    return [ModelArtifactRead.model_validate(m) for m in result.scalars().all()]


@router.get("/{artifact_id}", response_model=ModelArtifactRead)
async def get_model_artifact(artifact_id: str, db: DBSession) -> ModelArtifactRead:
    result = await db.execute(select(ModelArtifact).where(ModelArtifact.id == artifact_id))
    m = result.scalar_one_or_none()
    if not m:
        raise HTTPException(status_code=404, detail="ModelArtifact not found")
    return ModelArtifactRead.model_validate(m)
