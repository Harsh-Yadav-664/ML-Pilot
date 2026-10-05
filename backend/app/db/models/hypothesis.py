"""Hypothesis ORM model."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Text, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.experiment import Experiment


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="agent", nullable=False)  # agent, user
    status: Mapped[str] = mapped_column(String(50), default="proposed", nullable=False)
    agent_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="hypotheses")

    def __repr__(self) -> str:
        return f"<Hypothesis id={self.id!r} status={self.status!r}>"
