"""UI Adapter API endpoints for the Frontend."""
from __future__ import annotations

import os
import uuid
import logging
import shutil
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File

from app.api.deps import DBSession
from app.db.session import AsyncSessionLocal
from app.schemas.experiment import ExperimentCreate, ExperimentRead
from app.services.experiment_service import ExperimentService
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.profiling.profiler import DataProfiler
from ml.data.preparation.native.native_prep import NativeDataPreparationProvider
from ml.agents.decision_agent import DecisionAgent
from ai.gateway import AIGateway
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["ui-adapter"])

# For MVP, default if not provided
DEMO_PROJECT_ID = "demo-project-id"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/data/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV dataset and return its columns."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = CsvLoader()
        df = loader.load(file_path)
        columns = df.columns.tolist()
        return {
            "dataset_path": file_path,
            "filename": file.filename,
            "columns": columns,
            "total_rows": len(df)
        }
    except Exception as e:
        logger.error(f"Failed to parse uploaded CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


@router.get("/data/metrics")
async def get_data_metrics(
    dataset_path: str = "data.csv", 
    target_column: str = "target"
) -> dict[str, Any]:
    """Return dataset metrics for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(dataset_path)
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_column)
        
        return {
            "total_rows": profile.rows,
            "total_columns": profile.columns,
            "missing_data_percent": float(profile.missing_rate),
            "duplicate_rows": profile.duplicate_rows
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"total_rows": 0, "total_columns": 0, "missing_data_percent": 0, "duplicate_rows": 0}

@router.get("/data/leakage-warnings")
async def get_leakage_warnings(
    dataset_path: str = "data.csv", 
    target_column: str = "target"
) -> list[dict[str, Any]]:
    """Return leakage warnings for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(dataset_path)
        prep = NativeDataPreparationProvider()
        warnings = prep.detect_leakage(df, target_column=target_column)
        
        return [
            {
                "id": f"warn_{i}",
                "column": w.column,
                "message": w.reason,
                "severity": w.severity
            }
            for i, w in enumerate(warnings)
        ]
    except Exception as e:
        logger.error(f"Failed to get leakage: {e}")
        return []

@router.get("/agent/suggestions")
async def get_agent_suggestions(
    db: DBSession,
    dataset_path: str = "data.csv", 
    target_column: str = "target"
) -> list[dict[str, Any]]:
    """Trigger AI and return suggestions."""
    svc = ExperimentService(db)
    try:
        hypotheses = await svc.suggest_experiments(
            dataset_version=dataset_path,
            target_column=target_column,
            objective="Maximize F1 score while preventing overfitting",
            max_hypotheses=3
        )
        return hypotheses
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return []

async def background_runner(experiment_id: str):
    """Thin wrapper: ExperimentService.run_experiment_background opens its own session."""
    try:
        svc = ExperimentService(None)  # session unused — service opens its own
        await svc.run_experiment_background(experiment_id)
    except Exception as e:
        logger.error(f"Background task failed for experiment {experiment_id}: {e}")

@router.post("/experiments/run")
async def run_experiment(data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks) -> ExperimentRead:
    """Run an experiment based on a suggestion."""
    svc = ExperimentService(db)
    suggestion = data.get("feature_suggestion", {})
    dataset_path = data.get("dataset_path", "data.csv")
    target_column = data.get("target_column", "target")
    
    exp_create = ExperimentCreate(
        project_id=DEMO_PROJECT_ID,
        dataset_version=dataset_path,
        hypothesis=suggestion.get("reason", "No reason provided"),
        change_description=f"Added feature: {suggestion.get('name')} via formula {suggestion.get('formula')}",
        model_name="XGBClassifier",
        feature_set=[suggestion.get("name")] if suggestion.get("name") else [],
        parameters={
            "target_column": target_column,
            "feature_name": suggestion.get("name"),
            "formula": suggestion.get("formula")
        }
    )
    
    exp = await svc.create(exp_create)
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)

@router.get("/experiments/tree")
async def get_experiment_tree(db: DBSession) -> list[dict[str, Any]]:
    """Return all experiments formatted as a flat list."""
    svc = ExperimentService(db)
    exps, _ = await svc.list_by_project(DEMO_PROJECT_ID, page=1, page_size=100)
    
    result = []
    for e in exps:
        result.append({
            "id": e.id,
            "parent_id": e.parent_id,
            "model_name": e.model_name,
            "status": e.status,
            "metrics": e.metrics or {},
            "runtime_seconds": e.runtime_seconds or 0.0,
            "created_at": e.created_at.isoformat()
        })
    return result

# Auto-Optimize Agent
JOBS: dict[str, dict] = {}

async def _run_agent_task(job_id: str, dataset_path: str, target_column: str, n_hypotheses: int):
    gateway = AIGateway(settings)
    agent = DecisionAgent(gateway, settings)
    try:
        result = await agent.run_optimization_loop(dataset_path, target_column, n_hypotheses)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
    except Exception as e:
        logger.error(f"Auto-optimize failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@router.post("/agent/auto-optimize")
async def auto_optimize(
    data: dict[str, Any], 
    db: DBSession, 
    background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Trigger autonomous optimization loop."""
    dataset_path = data.get("dataset_path", "data.csv")
    target_column = data.get("target_column", "target")
    n_hypotheses = data.get("n_hypotheses", 5)
    
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}
    
    background_tasks.add_task(_run_agent_task, job_id, dataset_path, target_column, n_hypotheses)
    return {"job_id": job_id, "status": "running"}

@router.get("/agent/auto-optimize/{job_id}")
async def get_optimize_status(job_id: str) -> dict[str, Any]:
    """Poll the status of an auto-optimize job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
