"""Experiment Pydantic v2 schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.experiments.schema import ExperimentDecision, ExperimentStatus


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    project_id: str
    dataset_version: str
    hypothesis: str
    change_description: str
    model_name: str
    parameters: dict[str, Any] = {}
    validation_config: dict[str, Any] = {}
    feature_set: list[str] = []
    preprocessing_config: dict[str, Any] = {}
    budget: dict[str, float] = {}
    parent_id: str | None = None
    # Set from dataset_version when that is a stored version (ml/data/versions.py).
    data_version_id: str | None = None


class ExperimentRead(ExperimentCreate):
    id: str
    metrics: dict[str, float | None] | None = None
    artifacts: list[str] = []
    runtime_seconds: float | None = None
    cost_usd: float | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    decision: ExperimentDecision = ExperimentDecision.PENDING
    decision_reason: str | None = None
    agent_model: str | None = None
    manifest: dict[str, Any] | None = Field(
        None, description="What is needed to reproduce the run (RunManifest), once it completed"
    )
    created_at: datetime
    updated_at: datetime


class ExperimentUpdate(BaseModel):
    model_config = ConfigDict()

    status: ExperimentStatus | None = None
    decision: ExperimentDecision | None = None
    decision_reason: str | None = None
    metrics: dict[str, float | None] | None = None
