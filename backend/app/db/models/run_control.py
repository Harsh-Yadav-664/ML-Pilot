"""What a person tells a running relational run, and what the run asks them (#59)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RunCheckpoint(Base):
    """A question the loop is waiting on. It lives in the database, so a restart does not lose it."""

    __tablename__ = "run_checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id"), nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)  # approve_feature
    state: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending, approved, vetoed, timeout
    recommended: Mapped[str] = mapped_column(String(20), nullable=False)  # what a timeout does
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)  # what to look at
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    timeout_seconds: Mapped[float] = mapped_column(nullable=False, default=300.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RunSuggestion(Base):
    """A feature idea from the user. The proposer sees it next round; the same checks apply."""

    __tablename__ = "run_suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="new", nullable=False)  # new, used
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    used_in_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
