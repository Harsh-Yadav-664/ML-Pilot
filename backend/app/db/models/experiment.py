"""Experiment ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.hypothesis import Hypothesis
    from app.db.models.project import Project


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("experiments.id"), nullable=True
    )
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(50), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    change_description: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    validation_config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    preprocessing_config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    feature_set: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    budget: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifacts: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="created", nullable=False)
    decision: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("runs.id"), nullable=True)
    data_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("data_versions.id"), nullable=True
    )
    val_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    test_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # The full accept/reject record (rule, gains, evidence); `decision` stays the short label.
    decision_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # What is needed to reproduce the run (ml/experiments/manifest.py), written once
    # when it completes.
    manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision_mode: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # rule, llm, fallback
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="experiments")
    hypotheses: Mapped[list[Hypothesis]] = relationship(
        "Hypothesis", back_populates="experiment", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id!r} status={self.status!r}>"
