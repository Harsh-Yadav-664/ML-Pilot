"""Privacy settings of a project and the log of every prompt sent to an LLM (#48)."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.api.deps import DBSession, ProjectID
from app.core import datasets
from app.db.models.llm_call import LLMCall
from app.schemas.privacy import LLMCallDetail, LLMCallRead, PrivacyRead, PrivacySettings
from app.services.privacy_service import PrivacyService

router = APIRouter(prefix="/projects/{project_id}", tags=["privacy"])


@router.get("/privacy", response_model=PrivacyRead)
async def get_privacy(project_id: ProjectID, db: DBSession) -> PrivacyRead:
    return await PrivacyService(db).read(project_id)


@router.put("/privacy", response_model=PrivacyRead)
async def set_privacy(data: PrivacySettings, project_id: ProjectID, db: DBSession) -> PrivacyRead:
    """Replace the level and the never-send list. Takes effect on the next prompt."""
    return await PrivacyService(db).update(project_id, data)


@router.get("/llm-calls", response_model=list[LLMCallRead])
async def list_llm_calls(
    project_id: ProjectID,
    db: DBSession,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[LLMCall]:
    """Every prompt sent for this project, newest first (no prompt text: see the detail)."""
    result = await db.execute(
        select(LLMCall)
        .where(LLMCall.project_id == project_id)
        .order_by(LLMCall.created_at.desc(), LLMCall.id)
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars())


@router.get("/llm-calls/{call_id}", response_model=LLMCallDetail)
async def get_llm_call(call_id: str, project_id: ProjectID, db: DBSession) -> LLMCallDetail:
    """One call with the full text of the prompt, its manifest and the response."""
    row = await db.get(LLMCall, call_id)
    if row is None or row.project_id != project_id:
        raise HTTPException(404, "LLM call not found")
    if not row.prompt_path:
        raise HTTPException(404, "This call has no stored prompt")
    path = datasets.PROJECTS_DIR / row.prompt_path
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise HTTPException(500, f"The stored prompt could not be read: {e}") from e
    return LLMCallDetail(
        **LLMCallRead.model_validate(row, from_attributes=True).model_dump(),
        system=stored.get("system", ""),
        prompt=stored.get("prompt", ""),
        response=stored.get("response"),
        manifest=stored.get("manifest", {}),
    )
