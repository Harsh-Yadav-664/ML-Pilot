"""Chat panel: grounded Q&A over a project's recorded experiments."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ai.router import TaskType
from app.api.deps import DBSession, Gateway, ProjectID
from app.schemas.api import AskRequest, AskResponse
from app.services.experiment_service import ExperimentService
from app.services.privacy_service import PrivacyService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask_history(
    request: AskRequest, project_id: ProjectID, db: DBSession, gateway: Gateway
) -> AskResponse:
    """Answer a question strictly from the project's recorded experiment history."""
    exps, _ = await ExperimentService(db).list_by_project(project_id, 1, 50)
    history = [
        {
            "experiment": e.id,
            "model": e.model_name,
            "status": e.status,
            "metrics": e.metrics,
            "reason": e.decision_reason,
        }
        for e in exps
    ]
    prompt = (
        (await PrivacyService(db).builder(project_id))
        .prompt("chat.ask")
        .text("User asked", request.query)
        .facts("Experiment History", history)
        .text("", "Answer strictly based on the history above. Cite experiment IDs.")
        .build()
    )
    try:
        answer = await gateway.complete(TaskType.ANALYZE, prompt)
    except RuntimeError as e:
        logger.error(f"Failed to answer Q&A: {e}")
        raise HTTPException(status_code=502, detail=f"Q&A failed: {e}") from e
    return AskResponse(answer=answer, grounding_context=prompt.text)
