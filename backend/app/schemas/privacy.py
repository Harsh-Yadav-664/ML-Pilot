"""Privacy settings and the prompt log, as the API shows them."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ai.context_builder import PrivacyLevel


class PrivacySettings(BaseModel):
    level: PrivacyLevel = Field(
        PrivacyLevel.schema_and_stats, description="What the LLM may see of the data"
    )
    never_send: list[str] = Field(
        default_factory=list,
        description="Columns that are never sent: `column` (any table) or `table.column`",
    )


class PrivacyRead(PrivacySettings):
    level_summaries: dict[str, str] = Field(description="One line per level, for the settings page")
    warning: str | None = Field(None, description="Set when the level can send cell values")


class LLMCallRead(BaseModel):
    id: str
    created_at: datetime
    purpose: str
    provider: str
    model: str
    decision_mode: str = Field(description="llm, fallback (offline stub), failed or pending")
    privacy_level: str | None
    prompt_chars: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float | None
    error: str | None


class LLMCallDetail(LLMCallRead):
    system: str
    prompt: str
    response: Any = None
    manifest: dict[str, Any] = Field(default_factory=dict)
