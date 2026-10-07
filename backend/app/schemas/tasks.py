"""Request and response models for prediction task specs (#49)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.domain import RunStatus
from ml.tasks.feasibility import Thresholds


class SpecIssueRead(BaseModel):
    path: str = Field(
        description="Dotted path of the field, such as 'cutoffs.end' or 'eligibility[1]'"
    )
    message: str
    severity: Literal["error", "warning"]


class TaskSpecInput(BaseModel):
    yaml: str = Field(description="The task spec as YAML (docs/task_spec.md)")
    connection_id: str = Field(
        description="The connection whose schema the spec is checked against"
    )
    data_version_id: str | None = Field(
        None, description="A database version: its as_of is where the data ends. Default: now"
    )


class TaskSpecValidation(BaseModel):
    spec: dict[str, Any] | None = Field(None, description="The parsed spec, defaults filled in")
    issues: list[SpecIssueRead]
    schema_fingerprint: str | None = None


class TaskSpecRead(BaseModel):
    id: str
    project_id: str
    name: str
    version: int
    status: Literal["draft", "confirmed"]
    yaml: str
    spec: dict[str, Any]
    connection_id: str | None
    schema_fingerprint: str | None
    confirmed_by: str | None
    confirmed_at: datetime | None
    used_by_runs: int = Field(description="Runs that point at this exact version")
    created_at: datetime
    issues: list[SpecIssueRead] = Field(
        default_factory=list, description="Problems found when it was saved or confirmed"
    )


class ConfirmRequest(BaseModel):
    data_version_id: str | None = None


class LabelPreviewRequest(BaseModel):
    data_version_id: str | None = Field(
        None,
        description="A database snapshot to build the labels from (and write them into the "
        "project's DuckDB file), or a live version. Default: the live database as of now",
    )
    materialize: bool = Field(
        True, description="Snapshots only: also write the labels as a table in the project"
    )
    thresholds: Thresholds | None = Field(
        None, description="Limits for the feasibility checks. Default: the standard limits"
    )


class CutoffCountRead(BaseModel):
    cutoff: datetime
    window_end: datetime
    eligible: int = Field(description="Entities that are eligible at this cutoff")
    positives: int | None = Field(None, description="Binary tasks: entities with label 1")
    base_rate: float | None = Field(None, description="Binary tasks: positives / eligible")
    mean_label: float | None = Field(None, description="Regression tasks")


class DroppedCutoffRead(BaseModel):
    cutoff: datetime
    window_end: datetime
    reason: str


class FeasibilityCheckRead(BaseModel):
    code: str
    status: Literal["ok", "warn", "block"]
    message: str
    numbers: dict[str, Any]


class FeasibilityRead(BaseModel):
    status: Literal["ok", "warn", "block"] = Field(description="The worst of the checks")
    blocked: bool = Field(description="A run cannot start unless it is started with an override")
    reasons: list[str] = Field(description="The blockers, or if none, the warnings")
    checks: list[FeasibilityCheckRead]
    thresholds: Thresholds
    metric: str = Field(description="The metric the task will be judged by")
    suggested_metric: str | None = Field(None, description="Set when the base rate is very low")


class LabelPreview(BaseModel):
    sql: str = Field(description="The generated query, for reading and review")
    dialect: Literal["duckdb", "postgres"]
    mode: Literal["snapshot", "live"]
    as_of: datetime
    cutoffs: list[CutoffCountRead]
    dropped_cutoffs: list[DroppedCutoffRead] = Field(
        description="Cutoffs whose label window ends after the data does: left out"
    )
    total_rows: int
    table: str | None = Field(None, description="Where the labels were written, if they were")
    feasibility: FeasibilityRead
    note: str | None = None


class SplitPreviewRequest(LabelPreviewRequest):
    folds: int = Field(
        3, ge=2, le=10, description="Expanding-window folds over the training cutoffs"
    )


class TimelineEntry(BaseModel):
    cutoff: datetime
    window_end: datetime
    part: Literal["train", "val", "test", "purged_train", "purged_val"]
    rows: int = Field(description="Label rows (eligible entities) at this cutoff")


class FoldRead(BaseModel):
    n_train: int
    n_val: int
    val_from: datetime
    val_to: datetime


class SplitPreview(BaseModel):
    val_from: datetime
    test_from: datetime
    timeline: list[TimelineEntry]
    n_train: int
    n_val: int
    n_test: int
    n_purged_train: int = Field(description="Rows whose label window crosses val_from: left out")
    n_purged_val: int = Field(description="Rows whose label window crosses test_from: left out")
    max_train_window_end: datetime = Field(description="At or before val_from")
    max_val_window_end: datetime | None = Field(description="At or before test_from")
    folds: list[FoldRead]
    dropped_cutoffs: list[DroppedCutoffRead]
    note: str


class RunStartRequest(BaseModel):
    data_version_id: str | None = Field(
        None, description="The database version to build the labels from. Default: live, as of now"
    )
    thresholds: Thresholds | None = None
    override: bool = Field(
        False, description="Start although the feasibility checks block the task; it is recorded"
    )
    override_reason: str | None = Field(None, max_length=500)


class TaskRunRead(BaseModel):
    id: str
    project_id: str
    task_id: str
    data_version_id: str | None
    status: RunStatus
    split_plan: dict[str, Any]
    manifest: dict[str, Any] = Field(
        description="What the run was started with: task version, split, feasibility report, "
        "and the override if there was one"
    )
    created_at: datetime
