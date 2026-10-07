"""Prediction task specs of a project: validate, save as versions, confirm (#49)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import DBSession, Gateway, ProjectID
from app.schemas.tasks import (
    BaselineRead,
    ConfirmRequest,
    LabelPreview,
    LabelPreviewRequest,
    RunStartRequest,
    SplitPreview,
    SplitPreviewRequest,
    TaskDraft,
    TaskDraftRequest,
    TaskRunRead,
    TaskSpecInput,
    TaskSpecRead,
    TaskSpecValidation,
)
from app.services.baseline_service import BaselineService
from app.services.task_service import TaskService
from ml.tasks.spec import json_schema

router = APIRouter(prefix="/projects/{project_id}/tasks", tags=["tasks"])


@router.get("/schema")
async def get_task_spec_schema(project_id: ProjectID) -> dict[str, Any]:
    """JSON Schema of the task spec, for the editor and for structured LLM output."""
    return json_schema()


@router.post("/draft", response_model=TaskDraft)
async def draft_task_spec(
    data: TaskDraftRequest, project_id: ProjectID, db: DBSession, gateway: Gateway
) -> TaskDraft:
    """Turn a question such as "which customers will stop ordering in the next 30 days?" into a
    task spec to read and confirm. A question that is too vague gets ``status: clarify`` and a
    question back, never a guess. The answer says whether a language model or the offline
    rule-based drafter wrote it (``decision_mode``). Nothing is saved or run."""
    return await TaskService(db).draft(project_id, data, gateway)


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


@router.post("/{task_id}/preview-labels", response_model=LabelPreview)
async def preview_labels(
    task_id: str,
    project_id: ProjectID,
    db: DBSession,
    body: LabelPreviewRequest | None = None,
) -> LabelPreview:
    """Build the labels of a task at its cutoff dates: the generated SQL, eligible entities,
    positives and base rate per cutoff, and the cutoffs left out because their label window is
    not complete. With a snapshot the labels are also written to the project's DuckDB file;
    without one they are computed read-only on the database through the SQL guard."""
    return await TaskService(db).preview_labels(project_id, task_id, body or LabelPreviewRequest())


@router.post("/{task_id}/preview-split", response_model=SplitPreview)
async def preview_split(
    task_id: str,
    project_id: ProjectID,
    db: DBSession,
    body: SplitPreviewRequest | None = None,
) -> SplitPreview:
    """The temporal split of a task, by cutoff: which cutoffs train, validate and test, which
    rows are left out because their label window crosses a boundary, and the expanding-window
    folds over the training cutoffs. Relational tasks have no other kind of split."""
    return await TaskService(db).preview_split(project_id, task_id, body or SplitPreviewRequest())


@router.post("/{task_id}/runs", response_model=TaskRunRead, status_code=status.HTTP_201_CREATED)
async def start_run(
    task_id: str, project_id: ProjectID, db: DBSession, body: RunStartRequest | None = None
) -> TaskRunRead:
    """Record a run of a confirmed task. The feasibility checks run first: if they block the
    task (too few positives, say) the answer is a 422 with the reasons, unless the request says
    ``override: true``, in which case the run is recorded and its manifest keeps the report and
    the override. This records the run; it does not train a model yet."""
    return await TaskService(db).start_run(project_id, task_id, body or RunStartRequest())


@router.post("/{task_id}/runs/{run_id}/baseline", response_model=BaselineRead)
async def run_baseline(
    task_id: str, run_id: str, project_id: ProjectID, db: DBSession
) -> BaselineRead:
    """Build the baseline of a run: automatic aggregations over the tables related to the entity
    (counts and recency over several windows, sums and means of numeric columns, shares, counts
    per common category, attributes of the entity row), every one checked by the point-in-time
    guard; constant and duplicate features are dropped, then LightGBM is trained on the temporal
    split and scored on the validation rows. The result is recorded as the run's first champion.
    Needs a snapshot data version. The test rows are not scored here."""
    return await BaselineService(db).run(project_id, task_id, run_id)
