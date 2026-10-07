"""Prediction task specs of a project: validate, save as versions, confirm (#49)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import DBSession, ProjectID
from app.schemas.tasks import ConfirmRequest, TaskSpecInput, TaskSpecRead, TaskSpecValidation
from app.services.task_service import TaskService
from ml.tasks.spec import json_schema

router = APIRouter(prefix="/projects/{project_id}/tasks", tags=["tasks"])


@router.get("/schema")
async def get_task_spec_schema(project_id: ProjectID) -> dict[str, Any]:
    """JSON Schema of the task spec, for the editor and for structured LLM output."""
    return json_schema()


@router.post("/validate", response_model=TaskSpecValidation)
async def validate_task_spec(
    data: TaskSpecInput, project_id: ProjectID, db: DBSession
) -> TaskSpecValidation:
    """Check a spec against a connection's schema without saving it. Every problem is returned
    with the path of its field; warnings do not block confirming."""
    return await TaskService(db).validate(project_id, data)


@router.post("/", response_model=TaskSpecRead, status_code=status.HTTP_201_CREATED)
async def create_task_spec(
    data: TaskSpecInput, project_id: ProjectID, db: DBSession
) -> TaskSpecRead:
    """Save a new task as a draft (version 1). A spec that is not valid YAML or breaks the
    format is a 422; problems against the schema are returned in ``issues`` and block confirming."""
    return await TaskService(db).create(project_id, data)


@router.get("/", response_model=list[TaskSpecRead])
async def list_task_specs(
    project_id: ProjectID, db: DBSession, latest_only: bool = False
) -> list[TaskSpecRead]:
    return await TaskService(db).list_specs(project_id, latest_only)


@router.get("/{task_id}", response_model=TaskSpecRead)
async def get_task_spec(task_id: str, project_id: ProjectID, db: DBSession) -> TaskSpecRead:
    return await TaskService(db).get(project_id, task_id)


@router.put("/{task_id}", response_model=TaskSpecRead)
async def update_task_spec(
    task_id: str, data: TaskSpecInput, project_id: ProjectID, db: DBSession
) -> TaskSpecRead:
    """Edit a draft in place. Editing a confirmed spec saves the next version as a draft and
    leaves the confirmed one, and every run that uses it, untouched."""
    return await TaskService(db).update(project_id, task_id, data)


@router.post("/{task_id}/confirm", response_model=TaskSpecRead)
async def confirm_task_spec(
    task_id: str, project_id: ProjectID, db: DBSession, body: ConfirmRequest | None = None
) -> TaskSpecRead:
    """Confirm a draft: validated again, refused (422 with the issues) if any error remains.
    The schema fingerprint it was checked against is stored."""
    return await TaskService(db).confirm(
        project_id, task_id, body.data_version_id if body else None
    )
