"""Project ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.dataset import Dataset
    from app.db.models.experiment import Experiment
    from app.db.models.user import User


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # classification, regression
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    # Privacy level (#48), budgets and other per-project settings.
    settings: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=text("'{}'"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    owner: Mapped[User] = relationship("User", back_populates="projects")
    datasets: Mapped[list[Dataset]] = relationship(
        "Dataset", back_populates="project", lazy="select"
    )
    experiments: Mapped[list[Experiment]] = relationship(
        "Experiment", back_populates="project", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id!r} name={self.name!r}>"
