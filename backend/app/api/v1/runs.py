"""Relational runs (#58): start the loop, read where a run stands, list its features."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.api.deps import DBSession, ProjectID
from app.schemas.runs import (
    CheckpointDecision,
    CheckpointRead,
    RunFeaturesRead,
    RunLoopRequest,
    RunLoopStarted,
    RunNarration,
    RunStateRead,
    SettingsMessage,
    SettingsPreview,
    SuggestionCreate,
    SuggestionRead,
)
from app.services.bundle_service import BundleService, Dialect
from app.services.report_service import ReportFormat, ReportService
from app.services.run_service import RunService, SteerService

router = APIRouter(prefix="/projects/{project_id}/runs", tags=["runs"])


@router.post("/{run_id}/start", response_model=RunLoopStarted, status_code=202)
async def start_run(
    run_id: str, project_id: ProjectID, db: DBSession, body: RunLoopRequest | None = None
) -> RunLoopStarted:
    """Start the run as a background job: the baseline, then one proposed feature per round,
    each kept only if its paired gain on time-ordered folds of the training rows beats the
    margin, until a budget is reached, the rounds are used up or no feature has helped for
    ``patience`` rounds. The test rows are scored once, at the end. Needs a run made with
    ``POST .../tasks/{task}/runs`` on a snapshot. Cancel through the job."""
    return await RunService(db).start(project_id, run_id, body or RunLoopRequest())


@router.post("/{run_id}/resume", response_model=RunLoopStarted, status_code=202)
async def resume_run(run_id: str, project_id: ProjectID, db: DBSession) -> RunLoopStarted:
    """Resume an interrupted or cancelled run from its last champion. The run continues
    with its remaining rounds and budget. A run that was completed cannot be resumed."""
    return await RunService(db).resume(project_id, run_id)


@router.get("/{run_id}", response_model=RunStateRead)
async def get_run(run_id: str, project_id: ProjectID, db: DBSession) -> RunStateRead:
    return await RunService(db).state(project_id, run_id)


@router.get("/{run_id}/features", response_model=RunFeaturesRead)
async def get_run_features(run_id: str, project_id: ProjectID, db: DBSession) -> RunFeaturesRead:
    """Every feature of the run: the baseline's, and each proposal with its stage or gain."""
    return await RunService(db).features(project_id, run_id)


@router.get("/{run_id}/narration", response_model=RunNarration)
async def get_run_narration(run_id: str, project_id: ProjectID, db: DBSession) -> RunNarration:
    """The run's events in words. Each sentence is filled from its event, so every number in it
    is one the run recorded; the event is returned with it."""
    return await SteerService(db).narration(project_id, run_id)


@router.get("/{run_id}/checkpoints", response_model=list[CheckpointRead])
async def list_checkpoints(
    run_id: str, project_id: ProjectID, db: DBSession
) -> list[CheckpointRead]:
    """The questions the run asked: a pending one is waiting for an answer."""
    return await SteerService(db).checkpoints(project_id, run_id)


@router.post("/{run_id}/checkpoints/{checkpoint_id}", response_model=CheckpointRead)
async def answer_checkpoint(
    run_id: str,
    checkpoint_id: str,
    body: CheckpointDecision,
    project_id: ProjectID,
    db: DBSession,
) -> CheckpointRead:
    """Approve or veto the proposal the run is waiting on. A veto keeps the feature out of the
    model without testing it. Answered once: a second answer is a 409."""
    return await SteerService(db).answer(project_id, run_id, checkpoint_id, body)


@router.get("/{run_id}/suggestions", response_model=list[SuggestionRead])
async def list_suggestions(
    run_id: str, project_id: ProjectID, db: DBSession
) -> list[SuggestionRead]:
    return await SteerService(db).suggestions(project_id, run_id)


@router.post("/{run_id}/suggestions", response_model=SuggestionRead, status_code=201)
async def add_suggestion(
    run_id: str, body: SuggestionCreate, project_id: ProjectID, db: DBSession
) -> SuggestionRead:
    """Tell the run a feature idea. The model sees it in its next round; whatever it proposes
    is checked and tested like any other proposal."""
    return await SteerService(db).suggest(project_id, run_id, body)


@router.post("/{run_id}/settings", response_model=SettingsPreview)
async def preview_settings(
    run_id: str, body: SettingsMessage, project_id: ProjectID, db: DBSession
) -> SettingsPreview:
    """Read a sentence such as 'budget $1' and say what it would change. Changes nothing."""
    return await SteerService(db).preview_settings(project_id, run_id, body.message)


@router.post("/{run_id}/settings/apply", response_model=SettingsPreview)
async def apply_settings(
    run_id: str, body: SettingsMessage, project_id: ProjectID, db: DBSession
) -> SettingsPreview:
    """Apply what the sentence says to a queued or running run, after the person has seen the
    preview. The run reads its settings before every round."""
    return await SteerService(db).apply_settings(project_id, run_id, body.message)


REPORT_FILES = {"markdown": "md", "html": "html", "json": "json"}


@router.get(
    "/{run_id}/report",
    response_class=Response,
    responses={
        200: {
            "content": {"text/markdown": {}, "text/html": {}, "application/json": {}},
            "description": "The evidence report in the format asked for",
        }
    },
)
async def get_run_report(
    run_id: str,
    project_id: ProjectID,
    db: DBSession,
    format: Annotated[
        ReportFormat,
        Query(
            description="markdown, html (one file, no external requests) or json "
            "(the records and the number log)"
        ),
    ] = "markdown",
    download: Annotated[bool, Query(description="Send it as a file to save")] = False,
) -> Response:
    """The run's evidence report: the question, the data, the split, every feature with its gain
    and SQL, the safety findings, the limits, and the cost. Every number in it is read from what
    the run stored. Works while a run is going or after it ended, for what has been stored."""
    text, media_type = await ReportService(db).render(project_id, run_id, format)
    headers = (
        {
            "Content-Disposition": f'attachment; filename="report-{run_id[:8]}.{REPORT_FILES[format]}"'
        }
        if download
        else {}
    )
    return Response(text, media_type=media_type, headers=headers)


@router.get(
    "/{run_id}/export",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}, "description": "The export bundle"}},
)
async def export_run(
    run_id: str,
    project_id: ProjectID,
    db: DBSession,
    dialect: Annotated[
        Dialect | None,
        Query(description="SQL dialect of the features; default: the source's, else duckdb"),
    ] = None,
) -> Response:
    """A zip with the champion's features as SQL and a dbt project, the model in LightGBM text
    format, the task, the manifest, the report, and score.py, which scores from your database
    read-only without MLPilot. Only for a run that ended and kept its model."""
    data, filename = await BundleService(db).bundle(project_id, run_id, dialect)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
