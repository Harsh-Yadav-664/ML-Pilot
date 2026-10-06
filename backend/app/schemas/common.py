"""Shared Pydantic v2 schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class APIResponse[T](BaseModel):
    """Generic API response wrapper."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    data: T | None = None
    message: str | None = None


class PaginatedResponse[T](BaseModel):
    """Paginated list response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class ErrorDetail(BaseModel):
    model_config = ConfigDict()

    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict()

    success: bool = False
    error: ErrorDetail
    request_id: str | None = None
