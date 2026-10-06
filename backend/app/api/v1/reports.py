"""Reports API endpoints (Phase 0 stub)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportSummary(BaseModel):
    project_id: str
    message: str


@router.get("/project/{project_id}", response_model=ReportSummary)
async def get_project_report(project_id: str) -> ReportSummary:
    """Phase 0 stub — report generation implemented in Phase 2."""
    return ReportSummary(
        project_id=project_id,
        message="Report generation is available in Phase 2. This endpoint is a placeholder.",
    )
