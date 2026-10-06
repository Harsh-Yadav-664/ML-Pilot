"""Chat panel: grounded Q&A over a project's recorded experiments."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ai.router import TaskType
from app.api.deps import DBSession, Gateway, ProjectID
from app.schemas.api import AskRequest, AskResponse
from app.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask_history(
    request: AskRequest, project_id: ProjectID, db: DBSession, gateway: Gateway
) -> AskResponse:
    """Answer a question strictly from the project's recorded experiment history."""
    exps, _ = await ExperimentService(db).list_by_project(project_id, 1, 50)
    context = "\n".join(
        f"Exp {e.id}: Model {e.model_name}, Status {e.status}, Metrics: {e.metrics},"
        f" Reason: {e.decision_reason}"
        for e in exps
    )
    prompt = (
        f"User asked: {request.query}\n\nExperiment History:\n{context}\n\n"
        "Answer strictly based on the history above. Cite experiment IDs."
    )
    try:
        answer = await gateway.complete(TaskType.ANALYZE, prompt)
    except RuntimeError as e:
        logger.error(f"Failed to answer Q&A: {e}")
        raise HTTPException(status_code=502, detail=f"Q&A failed: {e}") from e
    return AskResponse(answer=answer, grounding_context=context)
