"""Request and response models of the resource API (`/api/v1/projects/...`).

These models are the contract with the frontend: the OpenAPI schema generated from
them is turned into `frontend/src/api/schema.d.ts`, and CI fails when the two drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Datasets (immutable data versions) -------------------------------------------------


class SampleDatasetRequest(BaseModel):
    dataset_name: str = "telecom_churn"


class SqlSnapshotRequest(BaseModel):
    """Run one read-only query and store the result as a data version.

    Name a saved connection (``connection_id``, see /connections) or, until the UI moves to
    those, pass a ``connection_string`` (never stored or echoed). Exactly one of the two.
    """

    connection_id: str | None = Field(None, description="A saved connection of this project")
    connection_string: str | None = Field(
        None, description="Legacy: read-only connection; never stored or echoed"
    )
    query: str


class DatasetInfo(BaseModel):
    """What loading data returns: the stored version and its columns."""

    data_version_id: str
    short_hash: str
    filename: str
    columns: list[str]
    total_rows: int
    default_target: str | None = None


class DataMetrics(BaseModel):
    total_rows: int
    total_columns: int
    missing_data_percent: float
    duplicate_rows: int


class ColumnProfile(BaseModel):
    name: str
    dtype: Literal["int", "float", "category", "datetime", "bool", "string"]
    role: Literal["feature", "target", "id"]
    missing_pct: float
    unique: int
    dist: list[float]


class LeakageFinding(BaseModel):
    id: str
    column: str
    message: str
    severity: Literal["low", "medium", "high"]
    category: str
    check: str
    evidence: dict[str, Any]


class FeatureSuggestion(BaseModel):
    name: str
    formula: str
    reason: str
    risk: str = ""
    required_columns: list[str] = Field(default_factory=list)
    availability_assumption: str = ""
    llm: dict[str, Any] | None = Field(
        None, description="Provider, model, cost and decision_mode of the call that proposed it"
    )


# ---- Experiments ------------------------------------------------------------------------


class SuggestionIn(BaseModel):
    name: str | None = None
    formula: str | None = None
    reason: str | None = None


class RunExperimentRequest(BaseModel):
    data_version_id: str
    target_column: str
    model_name: str | None = None
    parent_id: str | None = None
    feature_suggestion: SuggestionIn = Field(default_factory=SuggestionIn)


class DatasetTargetRequest(BaseModel):
    data_version_id: str
    target_column: str


class ExperimentNode(BaseModel):
    """One experiment as the UI's experiment tree shows it."""

    id: str
    parent_id: str | None
    data_version_id: str | None
    model_name: str
    status: str
    metrics: dict[str, float | None]
    runtime_seconds: float
    created_at: datetime
    title: str
    feature: str | None
    decision: Literal["keep", "reject", "baseline", "none"]
    error: str | None
    hypothesis_mode: str | None = Field(
        None, description="'fallback' when the idea came from the offline stub, not a real LLM"
    )
    on_champion_path: bool
    champion: bool


class ExportScript(BaseModel):
    experiment_id: str
    filename: str
    script: str


# ---- Agent jobs -------------------------------------------------------------------------


class AutoOptimizeRequest(BaseModel):
    data_version_id: str
    target_column: str
    n_hypotheses: int = Field(5, ge=1, le=20)


class JobRead(BaseModel):
    """A background job (app/jobs/runner.py)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    progress: float
    current_step: str | None
    cancel_requested: bool
    error: str | None
    result: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobEventRead(BaseModel):
    """One entry of a job's ordered event log."""

    model_config = ConfigDict(from_attributes=True)

    seq: int
    ts: datetime
    type: Literal["step", "proposal", "guard", "cv_result", "decision", "checkpoint", "log"]
    payload: dict[str, Any]


# ---- Chat -------------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    grounding_context: str


class Debrief(BaseModel):
    debrief: str
    feature_importances: dict[str, float]
    importance_method: str
