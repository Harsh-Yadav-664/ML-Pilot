"""TaskSpec ORM model: a versioned prediction task spec (entity, target, horizon, cutoffs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TaskSpec(Base):
    __tablename__ = "task_specs"
    __table_args__ = (UniqueConstraint("project_id", "name", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False, server_default="")
    # The connection whose schema graph the spec is validated against (never a secret).
    connection_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("connections.id"), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    yaml: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)  # confirmed
    confirmed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    schema_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # How the spec was drafted from a question (#53): question, decision_mode, assumptions, LLM calls.
    draft_source: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<TaskSpec id={self.id!r} version={self.version!r} status={self.status!r}>"
