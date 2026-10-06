"""Run ORM model: one end-to-end run of a task spec on a data version."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    task_spec_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("task_specs.id"), nullable=True
    )
    data_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("data_versions.id"), nullable=True
    )
    split_plan: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    engine: Mapped[str] = mapped_column(String(100), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="created", nullable=False)
    budget: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    budget_used: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # experiments.run_id points back here, so this side is created after both tables exist.
    champion_experiment_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("experiments.id", use_alter=True),
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Run id={self.id!r} status={self.status!r}>"
