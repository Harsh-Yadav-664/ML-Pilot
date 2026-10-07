"""Request and response models of a relational run (#58)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.domain import FeatureRead, RunStatus


class RunLoopRequest(BaseModel):
    """How long the run may go on. A limit that is not set does not apply."""

    max_rounds: int = Field(
        20, ge=1, le=200, description="Features the model is asked for, at most"
    )
    patience: int = Field(
        5,
        ge=1,
        le=200,
        description="Stop after this many rounds in a row without an accepted feature",
    )
    max_cost_usd: float | None = Field(
        None, gt=0, description="Stop when the language model has cost this much"
    )
    max_seconds: float | None = Field(None, gt=0, description="Stop after this much wall time")
    max_proposals: int | None = Field(None, ge=1, description="Stop after this many proposals")
    feature_timeout_seconds: float = Field(
        120.0, gt=0, le=3600, description="One feature's query may run this long"
    )


class RunLoopStarted(BaseModel):
    run_id: str
    job_id: str = Field(
        description="Poll or stream /jobs/{job_id}/events; cancel with the jobs API"
    )


class RunStateRead(BaseModel):
    """Where a run stands, from what the loop has recorded so far."""

    id: str
    task_id: str | None
    data_version_id: str | None
    status: RunStatus
    stop_reason: str | None = Field(
        None, description="max_rounds, patience, budget:cost, budget:time, budget:proposals, no_llm"
    )
    budget: dict[str, Any]
    budget_used: dict[str, Any]
    champion_experiment_id: str | None
    champion_val_metrics: dict[str, float] | None
    test_metrics: dict[str, float] | None = Field(
        None, description="Scored once, when the run ends. Empty while it runs and after a cancel"
    )
    test_error: str | None = None
    rounds: int = Field(description="Proposals made so far")
    accepted: int = Field(description="Of those, kept because they helped")
    error: str | None
    notes: list[str]
    started_at: datetime | None
    finished_at: datetime | None


class RunFeaturesRead(BaseModel):
    run_id: str
    features: list[FeatureRead]
