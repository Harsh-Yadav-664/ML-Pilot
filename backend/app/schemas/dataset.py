"""Dataset Pydantic v2 schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(default="v1")
    format: str = Field(default="csv")
    target_column: Optional[str] = None


class DatasetCreate(DatasetBase):
    project_id: str


class DatasetRead(DatasetBase):
    id: str
    project_id: str
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    profile: Optional[dict[str, Any]] = None
    status: str
    file_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DatasetUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: Optional[str] = None
    target_column: Optional[str] = None
    status: Optional[str] = None
