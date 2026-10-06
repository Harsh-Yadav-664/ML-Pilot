"""Dataset Pydantic v2 schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(default="v1")
    format: str = Field(default="csv")
    target_column: str | None = None


class DatasetCreate(DatasetBase):
    project_id: str


class DatasetRead(DatasetBase):
    id: str
    project_id: str
    num_rows: int | None = None
    num_columns: int | None = None
    profile: dict[str, Any] | None = None
    status: str
    file_path: str | None = None
    created_at: datetime
    updated_at: datetime


class DatasetUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str | None = None
    target_column: str | None = None
    status: str | None = None
