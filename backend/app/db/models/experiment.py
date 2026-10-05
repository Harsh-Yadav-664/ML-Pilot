"""Experiment ORM model."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Text, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.project import Project
    from app.db.models.hypothesis import Hypothesis


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=True)
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
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    artifacts: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    runtime_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="created", nullable=False)
    decision: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    agent_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="experiments")
    hypotheses: Mapped[list["Hypothesis"]] = relationship("Hypothesis", back_populates="experiment", lazy="select")

    def __repr__(self) -> str:
        return f"<Experiment id={self.id!r} status={self.status!r}>"
