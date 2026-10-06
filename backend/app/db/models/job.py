"""Job ORM models: durable background work (app/jobs/runner.py) and its event log."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Job(Base):
    """One unit of background work. Status: queued, running, succeeded, failed, cancelled."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("runs.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)  # experiment, auto_optimize
    params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0..1
    current_step: Mapped[str | None] = mapped_column(String(200), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} kind={self.kind!r} status={self.status!r}>"


class JobEvent(Base):
    """What a job did, in order. The single source for progress, narration and reports."""

    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    # step, proposal, guard, cv_result, decision, checkpoint, log
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
