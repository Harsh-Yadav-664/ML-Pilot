"""Hypothesis Pydantic v2 schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HypothesisBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    text: str = Field(..., min_length=1)
    rationale: Optional[str] = None
    source: str = Field(default="agent")
    tags: list[str] = []


class HypothesisCreate(HypothesisBase):
    experiment_id: str
    project_id: str
    agent_model: Optional[str] = None


class HypothesisRead(HypothesisBase):
    id: str
    experiment_id: str
    project_id: str
    status: str
    agent_model: Optional[str] = None
    created_at: datetime


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: Optional[str] = None
    rationale: Optional[str] = None
