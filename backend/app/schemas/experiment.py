"""Experiment Pydantic v2 schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ml.experiments.schema import ExperimentStatus, ExperimentDecision


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
    parent_id: Optional[str] = None


class ExperimentRead(ExperimentCreate):
    id: str
    metrics: Optional[dict[str, Optional[float]]] = None
    artifacts: list[str] = []
    runtime_seconds: Optional[float] = None
    cost_usd: Optional[float] = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    decision: ExperimentDecision = ExperimentDecision.PENDING
    decision_reason: Optional[str] = None
    agent_model: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ExperimentUpdate(BaseModel):
    model_config = ConfigDict()

    status: Optional[ExperimentStatus] = None
    decision: Optional[ExperimentDecision] = None
    decision_reason: Optional[str] = None
    metrics: Optional[dict[str, Optional[float]]] = None


class ExperimentSuggestRequest(BaseModel):
    dataset_version: str
    target_column: str
    objective: str
    max_hypotheses: int = 3


class ExperimentSuggestion(BaseModel):
    name: str
    formula: str
    reason: str
    risk: str
    required_columns: list[str]
    availability_assumption: str


class ExperimentSuggestResponse(BaseModel):
    dataset_version: str
    objective: str
    hypotheses: list[ExperimentSuggestion]
