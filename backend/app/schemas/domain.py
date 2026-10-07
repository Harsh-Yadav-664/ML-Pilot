"""Pydantic read/write schemas for the relational-product domain tables (#89).

Create schemas carry what a caller supplies; Read schemas add server-set fields and
are built from ORM rows (``from_attributes``). No endpoints use them yet; later issues do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DataVersionKind = Literal["file", "db_snapshot", "db_live"]
TaskSpecStatus = Literal["draft", "confirmed"]
RunStatus = Literal["created", "queued", "running", "completed", "failed", "cancelled", "stopped"]
FeatureKind = Literal["dfs", "llm_sql", "formula", "user"]
FeatureStatus = Literal[
    "proposed", "rejected_guard", "rejected_duplicate", "rejected_gain", "accepted"
]
DecisionMode = Literal["llm", "fallback", "rule"]


class _Read(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DataVersionCreate(BaseModel):
    id: str = Field(..., max_length=64, description="Content hash of the data")
    project_id: str
    kind: DataVersionKind
    source: dict[str, Any] = Field(
        default_factory=dict, description="Where it came from; no secrets"
    )
    n_rows: int | None = None
    n_columns: int | None = None
    max_event_time: dict[str, Any] | None = None


class DataVersionRead(DataVersionCreate, _Read):
    created_at: datetime


class ConnectionCreate(BaseModel):
    project_id: str
    name: str
    dialect: str
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    secret_ref: str | None = Field(
        None, description="Env var name or secret id, never the password"
    )
    ssl: dict[str, Any] | None = None


class ConnectionRead(ConnectionCreate, _Read):
    id: str
    last_tested_at: datetime | None = None
    can_write: bool | None = None
    created_at: datetime


class TaskSpecCreate(BaseModel):
    project_id: str
    version: int = 1
    yaml: str
    schema_fingerprint: str | None = None


class TaskSpecRead(TaskSpecCreate, _Read):
    id: str
    status: TaskSpecStatus
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None
    created_at: datetime


class RunCreate(BaseModel):
    project_id: str
    task_spec_id: str | None = None
    data_version_id: str | None = None
    split_plan: dict[str, Any] = Field(default_factory=dict)
    engine: str
    seed: int = 42
    budget: dict[str, Any] = Field(default_factory=dict)


class RunRead(RunCreate, _Read):
    id: str
    status: RunStatus
    budget_used: dict[str, Any] = Field(default_factory=dict)
    champion_experiment_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    created_at: datetime


class FeatureCreate(BaseModel):
    run_id: str
    name: str
    kind: FeatureKind
    sql: str | None = None
    formula: str | None = None
    ir: dict[str, Any] | None = None
    rationale: str | None = None


class FeatureRead(FeatureCreate, _Read):
    id: str
    status: FeatureStatus
    guard_results: dict[str, Any] | None = None
    gain: dict[str, Any] | None = None
    created_at: datetime


class LLMCallCreate(BaseModel):
    run_id: str | None = None
    purpose: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    decision_mode: Literal["llm", "fallback"]
    prompt_hash: str
    prompt_path: str | None = None


class LLMCallRead(LLMCallCreate, _Read):
    id: str
    created_at: datetime
