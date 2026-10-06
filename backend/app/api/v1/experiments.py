"""Experiments of a project: run, baseline, AI clean, tree, export and debrief."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ai.router import TaskType
from app.api.deps import DBSession, Gateway, ProjectID
from app.db.models import Experiment
from app.schemas.api import (
    DatasetTargetRequest,
    Debrief,
    ExperimentNode,
    ExportScript,
    RunExperimentRequest,
)
from app.schemas.experiment import ExperimentCreate, ExperimentRead
from app.services import data_service
from app.services.experiment_service import ExperimentService, champion_path
from app.services.export_service import ExportNotAvailable, training_script
from ml.models.engines import DEFAULT_ENGINE, EngineNotAvailable, get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/experiments", tags=["experiments"])

# A project's experiment tree is small; this bounds one response.
TREE_LIMIT = 500


async def background_runner(experiment_id: str) -> None:
    """Thin wrapper: ExperimentService.run_experiment_background opens its own session."""
    try:
        await ExperimentService(None).run_experiment_background(experiment_id)
    except Exception:  # last-resort guard; the service already marks the experiment failed
        logger.exception(f"Background task failed for experiment {experiment_id}")


async def _queue(
    db: DBSession, background_tasks: BackgroundTasks, data: ExperimentCreate
) -> ExperimentRead:
    exp = await ExperimentService(db).create(data)
    # Commit before scheduling: the background run reads the row from its own session.
    await db.commit()
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)


async def _project_experiment(db: DBSession, project_id: str, experiment_id: str) -> Experiment:
    exp = await ExperimentService(db).get(experiment_id)
    if exp is None or exp.project_id != project_id:
        raise HTTPException(status_code=404, detail="Experiment not found in this project")
    return exp


def _node(e: Experiment, path_ids: list[str]) -> ExperimentNode:
    params = e.parameters or {}
    if e.decision in ("keep", "reject"):
        decision = e.decision
    else:
        decision = "baseline" if not e.parent_id else "none"
    return ExperimentNode(
        id=e.id,
        parent_id=e.parent_id,
        data_version_id=e.data_version_id,
        model_name=e.model_name,
        status=e.status,
        metrics=e.metrics or {},
        runtime_seconds=e.runtime_seconds or 0.0,
        created_at=e.created_at,
        title=e.change_description,
        feature=params.get("feature_name"),
        decision=decision,
        error=e.decision_reason if e.status in ("failed", "rejected_invalid") else None,
        hypothesis_mode=(params.get("hypothesis_llm") or {}).get("decision_mode"),
        on_champion_path=e.id in path_ids,
        champion=bool(path_ids) and e.id == path_ids[-1],
    )


async def _experiments(
    db: DBSession, project_id: str, data_version_id: str | None
) -> list[Experiment]:
    exps, _ = await ExperimentService(db).list_by_project(project_id, page=1, page_size=TREE_LIMIT)
    if data_version_id:
        exps = [e for e in exps if e.data_version_id == data_version_id]
    return exps


@router.get("", response_model=list[ExperimentNode])
async def experiment_tree(
    project_id: ProjectID, db: DBSession, data_version_id: str | None = None
) -> list[ExperimentNode]:
    """The project's experiments as a flat tree, optionally only those on one data version."""
    exps = await _experiments(db, project_id, data_version_id)
    path_ids = [e.id for e in champion_path(exps)]
    return [_node(e, path_ids) for e in exps]


@router.post("", response_model=ExperimentRead)
async def run_experiment(
    request: RunExperimentRequest,
    project_id: ProjectID,
    db: DBSession,
    background_tasks: BackgroundTasks,
) -> ExperimentRead:
    """Queue an experiment that adds one suggested feature."""
    path = await data_service.version_path(db, request.data_version_id)
    model_name = request.model_name or DEFAULT_ENGINE
    try:
        get_engine(model_name)
    except EngineNotAvailable as e:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {e}") from e
    if request.parent_id:
        await _project_experiment(db, project_id, request.parent_id)
    s = request.feature_suggestion
    data = ExperimentCreate(
        project_id=project_id,
        parent_id=request.parent_id,
        dataset_version=str(path),
        hypothesis=s.reason or "No reason provided",
        change_description=f"Added feature: {s.name} via formula {s.formula}",
        model_name=model_name,
        feature_set=[s.name] if s.name else [],
        parameters={
            "target_column": request.target_column,
            "feature_name": s.name,
            "formula": s.formula,
        },
    )
    return await _queue(db, background_tasks, data)


@router.post("/baseline", response_model=ExperimentRead)
async def run_baseline(
    request: DatasetTargetRequest,
    project_id: ProjectID,
    db: DBSession,
    background_tasks: BackgroundTasks,
) -> ExperimentRead:
    """Queue a deterministic baseline without new features."""
    path = await data_service.version_path(db, request.data_version_id)
    data = ExperimentCreate(
        project_id=project_id,
        dataset_version=str(path),
        hypothesis="Deterministic baseline without new features",
        change_description="Baseline run using RandomForestClassifier",
        model_name="RandomForestClassifier",
        feature_set=[],
        parameters={
            "target_column": request.target_column,
            "model_params": {"n_estimators": 50, "random_state": 42},
        },
    )
    return await _queue(db, background_tasks, data)


@router.post("/auto-clean", response_model=ExperimentRead)
async def run_auto_clean(
    request: DatasetTargetRequest,
    project_id: ProjectID,
    db: DBSession,
    background_tasks: BackgroundTasks,
    gateway: Gateway,
) -> ExperimentRead:
    """Ask the LLM for a cleaning configuration and queue it as a baseline."""
    from ml.agents.cleaning_agent import CleaningStrategyError, DataCleaningAgent
    from ml.data.profiling.profiler import DataProfiler

    path = await data_service.version_path(db, request.data_version_id)
    df = data_service.load(path)
    profile = DataProfiler().profile(df, target_column=request.target_column)
    try:
        prep_config = await DataCleaningAgent(gateway).generate_cleaning_strategy(
            profile, request.target_column
        )
    except CleaningStrategyError as e:
        raise HTTPException(status_code=502, detail=f"AI cleaning strategy failed: {e}") from e
    data = ExperimentCreate(
        project_id=project_id,
        dataset_version=str(path),
        hypothesis="AI-driven advanced data cleaning (Robust Imputation, Encoding, Transforms)",
        change_description="AI Auto-Clean Configuration applied.",
        model_name="RandomForestClassifier",
        feature_set=[],
        preprocessing_config=prep_config,
        parameters={
            "target_column": request.target_column,
            "model_params": {"n_estimators": 50, "random_state": 42},
        },
    )
    return await _queue(db, background_tasks, data)


@router.get("/champion/export", response_model=ExportScript)
async def export_champion(
    project_id: ProjectID, data_version_id: str, db: DBSession
) -> ExportScript:
    """Training script of the current champion on one data version."""
    path = champion_path(await _experiments(db, project_id, data_version_id))
    if not path:
        raise HTTPException(
            status_code=404, detail="No completed baseline for this data version yet"
        )
    return _export(path[-1])


@router.get("/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(
    project_id: ProjectID, experiment_id: str, db: DBSession
) -> ExperimentRead:
    return ExperimentRead.model_validate(await _project_experiment(db, project_id, experiment_id))


@router.get("/{experiment_id}/export", response_model=ExportScript)
async def export_experiment(
    project_id: ProjectID, experiment_id: str, db: DBSession
) -> ExportScript:
    """A standalone Python script that rebuilds the experiment's features and model."""
    return _export(await _project_experiment(db, project_id, experiment_id))


def _export(exp: Experiment) -> ExportScript:
    try:
        script = training_script(exp)
    except ExportNotAvailable as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return ExportScript(experiment_id=exp.id, filename=f"train_{exp.id}.py", script=script)


@router.get("/{experiment_id}/debrief", response_model=Debrief)
async def debrief(
    project_id: ProjectID, experiment_id: str, db: DBSession, gateway: Gateway
) -> Debrief:
    """Plain-language debrief grounded in the recorded metrics and built-in importances.

    SHAP is not implemented yet.
    """
    exp = await _project_experiment(db, project_id, experiment_id)
    params = exp.parameters or {}
    importances = params.get("feature_importances") or {}
    method = params.get("importance_method", "not available: the run recorded no importances")
    if importances:
        drivers = f"Feature importances ({method}): {importances}."
    else:
        drivers = f"Feature importances are {method}. Do not name any feature as a driver."
    prompt = (
        f"Experiment {exp.id} used {exp.model_name}. Metrics: {exp.metrics}. {drivers} "
        "Write a 3-sentence plain-language debrief. Only mention columns listed above, "
        "and say that the importances are the model's built-in ones, not SHAP."
    )
    try:
        text = await gateway.complete(TaskType.REPORT, prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Debrief failed: {e}") from e
    return Debrief(debrief=text, feature_importances=importances, importance_method=method)
