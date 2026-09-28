"""UI Adapter API endpoints for the Frontend."""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.api.deps import DBSession
from app.db.session import AsyncSessionLocal
from app.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentSuggestRequest
from app.services.experiment_service import ExperimentService
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.profiling.profiler import DataProfiler
from ml.data.preparation.native.native_prep import NativeDataPreparationProvider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["ui-adapter"])

# For MVP, we hardcode the dataset path
DEMO_DATASET = "data.csv"
DEMO_TARGET = "target"
DEMO_PROJECT_ID = "demo-project-id"

@router.get("/data/metrics")
async def get_data_metrics() -> dict[str, Any]:
    """Return dataset metrics for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(DEMO_DATASET)
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=DEMO_TARGET)
        
        return {
            "total_rows": profile.rows,
            "total_columns": profile.columns,
            "missing_data_percent": round(profile.missing_rate * 100, 2),
            "duplicate_rows": profile.duplicate_rows
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        # Return graceful defaults if no data
        return {"total_rows": 0, "total_columns": 0, "missing_data_percent": 0, "duplicate_rows": 0}

@router.get("/data/leakage-warnings")
async def get_leakage_warnings() -> list[dict[str, Any]]:
    """Return leakage warnings for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(DEMO_DATASET)
        prep = NativeDataPreparationProvider()
        warnings = prep.detect_leakage(df, target_column=DEMO_TARGET)
        
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
async def get_agent_suggestions(db: DBSession) -> list[dict[str, Any]]:
    """Trigger AI and return suggestions."""
    svc = ExperimentService(db)
    try:
        hypotheses = await svc.suggest_experiments(
            dataset_version=DEMO_DATASET,
            target_column=DEMO_TARGET,
            objective="Maximize F1 score while preventing overfitting",
            max_hypotheses=3
        )
        return hypotheses
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return []

async def background_runner(experiment_id: str):
    async with AsyncSessionLocal() as session:
        try:
            svc = ExperimentService(session)
            await svc.run_experiment_background(experiment_id)
        except Exception as e:
            logger.error(f"Background task failed: {e}")

@router.post("/experiments/run")
async def run_experiment(data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks) -> ExperimentRead:
    """Run an experiment based on a suggestion."""
    svc = ExperimentService(db)
    suggestion = data.get("feature_suggestion", {})
    
    # Create the experiment record
    exp_create = ExperimentCreate(
        project_id=DEMO_PROJECT_ID,
        dataset_version=DEMO_DATASET,
        hypothesis=suggestion.get("reason", "No reason provided"),
        change_description=f"Added feature: {suggestion.get('name')} via formula {suggestion.get('formula')}",
        model_name="XGBClassifier",
        feature_set=[suggestion.get("name")] if suggestion.get("name") else [],
    )
    
    exp = await svc.create(exp_create)
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)

@router.get("/experiments/tree")
async def get_experiment_tree(db: DBSession) -> list[dict[str, Any]]:
    """Return all experiments formatted as a flat list (UI tree handles nesting if needed, or just renders flat for now)."""
    svc = ExperimentService(db)
    exps, _ = await svc.list_by_project(DEMO_PROJECT_ID, page=1, page_size=100)
    
    # Format to match UI expectations exactly
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
