"""Request and response models for prediction task specs (#49)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
