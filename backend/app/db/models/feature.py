"""Feature ORM model: one proposed feature, its definition, guard results and validated gain."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # dfs, llm_sql, formula, user
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    formula: Mapped[str | None] = mapped_column(Text, nullable=True)
    ir: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Feature IR (#100)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    # proposed, rejected_guard, rejected_duplicate, rejected_gain, accepted, vetoed
    status: Mapped[str] = mapped_column(String(30), default="proposed", nullable=False)
    guard_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    gain: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Feature id={self.id!r} name={self.name!r} status={self.status!r}>"
