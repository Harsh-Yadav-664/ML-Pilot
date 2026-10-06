"""Experiment Pydantic models — PRD-compliant schema."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # The candidate formula failed validation; nothing was trained.
    REJECTED_INVALID = "rejected_invalid"


class ExperimentDecision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"
    PENDING = "pending"


class ExperimentSpec(BaseModel):
    """Specification for a single ML experiment."""

    id: str = Field(..., description="Unique experiment identifier")
    parent_id: Optional[str] = Field(None, description="Parent experiment ID for lineage tracking")
    project_id: str = Field(..., description="Project this experiment belongs to")
    dataset_version: str = Field(..., description="Dataset version used")
    hypothesis: str = Field(..., description="The hypothesis being tested")
    change_description: str = Field(..., description="Description of what changed vs parent")
    model_name: str = Field(..., description="Name of the ML model to train")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    validation_config: dict[str, Any] = Field(default_factory=dict, description="Validation strategy config")
    feature_set: list[str] = Field(default_factory=list, description="Feature columns to use")
    preprocessing_config: dict[str, Any] = Field(default_factory=dict, description="Preprocessing pipeline config")
    budget: dict[str, float] = Field(
        default_factory=dict,
        description="Resource constraints: max_runtime_seconds, max_cost_usd"
    )


class ExperimentResult(BaseModel):
    """Result of a completed ML experiment."""

    id: str
    parent_id: Optional[str] = None
    project_id: str
    dataset_version: str
    hypothesis: str
    change_description: str
    model_name: str
    parameters: dict[str, Any]
    validation_config: dict[str, Any]
    metrics: dict[str, float] = Field(
        ..., description="Computed metrics e.g. {'f1': 0.821, 'recall': 0.847, 'auc': 0.91}"
    )
    artifacts: list[str] = Field(default_factory=list, description="Artifact file paths")
    runtime_seconds: float
    cost_usd: float
    status: ExperimentStatus
    decision: ExperimentDecision = ExperimentDecision.PENDING
    decision_reason: Optional[str] = None
    agent_model: Optional[str] = Field(None, description="LLM model that made the decision")
    timestamp: datetime
    parent_metrics: Optional[dict[str, float]] = Field(
        None, description="Parent experiment metrics for delta comparison"
    )
