"""DataVersion ORM model: an immutable snapshot of a file or database, identified by content hash."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DataVersion(Base):
    __tablename__ = "data_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # content hash
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # file, db_snapshot
    # Where the data came from (path, tables, query). Never contains secrets.
    source: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_columns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_event_time: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # per table
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<DataVersion id={self.id!r} kind={self.kind!r}>"
