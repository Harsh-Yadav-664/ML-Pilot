"""Request and response models of a relational run (#58)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    rollouts: int = Field(
        1,
        ge=1,
        le=10,
        description="Number of independent proposal histories to run. The best champion on validation is picked.",
    )
    approval_mode: Literal["auto", "confirm_task", "approve_each_feature"] = Field(
        "confirm_task",
        description="approve_each_feature: the run waits for you to approve or veto every "
        "proposal that passed the checks. The other two never pause it: the task spec was "
        "confirmed before the run",
    )
    checkpoint_timeout_seconds: float = Field(
        300.0,
        gt=0,
        le=86400,
        description="How long a question waits; with no answer the recommended action (approve) "
        "is taken and the run goes on",
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
    task_name: str | None = Field(None, description="The confirmed task's name")
    question: str | None = Field(
        None, description="The question the task was drafted from, if it was drafted from one"
    )
    data_version_id: str | None
    as_of: datetime | None = Field(None, description="Where the data of the run ends")
    split_plan: dict[str, Any] = Field(default_factory=dict, description="val_from and test_from")
    feasibility: dict[str, Any] | None = Field(
        None, description="The feasibility report the run was started with"
    )
    job_id: str | None = Field(None, description="The latest job of the run; cancel it there")
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


class RunFeatureRead(FeatureRead):
    description: str | None = Field(
        None, description="The feature in one English sentence: from its spec, else its rationale"
    )


class RunFeaturesRead(BaseModel):
    run_id: str
    features: list[RunFeatureRead]


class CheckpointRead(BaseModel):
    id: str
    run_id: str
    round: int
    kind: str
    state: Literal["pending", "approved", "vetoed", "timeout"]
    recommended: Literal["approve", "veto"] = Field(description="What a timeout does")
    payload: dict[str, Any] = Field(description="What to look at: the proposal and its query")
    note: str | None = None
    timeout_seconds: float
    created_at: datetime
    decided_at: datetime | None = None


class CheckpointDecision(BaseModel):
    decision: Literal["approve", "veto"]
    note: str | None = Field(None, max_length=500)


class SuggestionCreate(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=500,
        description="A feature idea in plain words. The model sees it in the next round; what it "
        "proposes from it goes through the same checks as any proposal",
    )


class SuggestionRead(BaseModel):
    id: str
    run_id: str
    text: str
    state: Literal["new", "used"]
    created_at: datetime
    used_in_round: int | None = None


class SettingsMessage(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=300,
        description="For example 'budget $1', 'at most 5 rounds', 'ask me before each feature'",
    )


class SettingChange(BaseModel):
    setting: str
    from_: Any = Field(alias="from")
    to: Any

    model_config = {"populate_by_name": True}


class SettingsPreview(BaseModel):
    """What a sentence would change. Nothing is changed until it is applied."""

    changes: list[SettingChange]
    unrecognised: list[str] = Field(description="Parts of the message that were not understood")
    summary: str
    applied: bool = Field(description="False for a preview; True once it was applied to the run")


class NarrationItem(BaseModel):
    seq: int
    type: str
    text: str
    at: datetime
    payload: dict[str, Any] = Field(description="The event the sentence was written from")


class RunNarration(BaseModel):
    run_id: str
    items: list[NarrationItem]
