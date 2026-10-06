"""Cached column statistics of a connected table (#96). Aggregates only, never rows."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TableStatsCache(Base):
    __tablename__ = "table_stats"
    __table_args__ = (UniqueConstraint("connection_id", "table_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    connection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("connections.id"), nullable=False
    )
    table_key: Mapped[str] = mapped_column(String(512), nullable=False)
    # What the statistics were computed against: size and mtime of a database file; 'live' for
    # Postgres, whose statistics stay until refreshed (data versions replace this in #97).
    cache_key: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
