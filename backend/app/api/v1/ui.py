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
    target_column: str = "target",
    user_query: Optional[str] = None
) -> list[dict[str, Any]]:
    """Trigger AI and return suggestions."""
    svc = ExperimentService(db)
    
    objective = "Maximize F1 score while preventing overfitting"
    if user_query:
        objective += f". User strictly requested: '{user_query}'. Generate features reflecting this."
        
    try:
        hypotheses = await svc.suggest_experiments(
            dataset_version=dataset_path,
            target_column=target_column,
            objective=objective,
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

@router.post("/experiments/baseline")
async def run_baseline(data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks) -> ExperimentRead:
    """Run a deterministic baseline experiment."""
    svc = ExperimentService(db)
    dataset_path = data.get("dataset_path", "data.csv")
    target_column = data.get("target_column", "target")
    
    exp_create = ExperimentCreate(
        project_id=DEMO_PROJECT_ID,
        dataset_version=dataset_path,
        hypothesis="Deterministic baseline without new features",
        change_description="Baseline run using RandomForestClassifier",
        model_name="RandomForestClassifier",
        feature_set=[],
        parameters={
            "target_column": target_column,
            "model_params": {"n_estimators": 50, "random_state": 42}
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

@router.post("/agent/auto-clean")
async def auto_clean_dataset(
    data: dict[str, Any], 
    db: DBSession,
    background_tasks: BackgroundTasks
) -> ExperimentRead:
    """Uses AI to generate an advanced cleaning strategy and runs it as a baseline."""
    dataset_path = data.get("dataset_path", "data.csv")
    target_column = data.get("target_column", "target")
    
    # 1. Profile Data
    loader = CsvLoader()
    df = loader.load(dataset_path)
    profiler = DataProfiler()
    profile = profiler.profile(df, target_column=target_column)
    
    # 2. Get AI Strategy
    gateway = AIGateway(settings)
    from ml.agents.cleaning_agent import DataCleaningAgent
    agent = DataCleaningAgent(gateway)
    prep_config = await agent.generate_cleaning_strategy(profile, target_column)
    
    # 3. Create a clean baseline experiment
    svc = ExperimentService(db)
    exp_create = ExperimentCreate(
        project_id=DEMO_PROJECT_ID,
        dataset_version=dataset_path,
        hypothesis="AI-driven advanced data cleaning (Robust Imputation, Encoding, Transforms)",
        change_description="AI Auto-Clean Configuration applied.",
        model_name="RandomForestClassifier",
        feature_set=[],
        preprocessing_config=prep_config,
        parameters={
            "target_column": target_column,
            "model_params": {"n_estimators": 50, "random_state": 42}
        }
    )
    
    exp = await svc.create(exp_create)
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)

@router.get("/experiments/{experiment_id}/export")
async def export_experiment_script(experiment_id: str, db: DBSession):
    """Export a python training script for the given experiment."""
    svc = ExperimentService(db)
    exp = await svc.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    script = f"""# MLPilot Auto-Generated Training Script
# Experiment ID: {exp.id}
# Hypothesis: {exp.hypothesis}

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Install dependencies if needed: pip install scikit-learn pandas xgboost lightgbm

import ast

def load_and_prepare_data():
    df = pd.read_csv("{exp.dataset_version}")
    
    # Feature Engineering
    feature_name = "{exp.parameters.get('feature_name', '')}"
    formula = "{exp.parameters.get('formula', '')}"
    if feature_name and formula:
        def _safe_eval(node):
            if isinstance(node, ast.Expression):
                return _safe_eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                if node.id in df.columns:
                    return df[node.id]
                raise ValueError(f"Column '{{node.id}}' not found")
            elif isinstance(node, ast.BinOp):
                left = _safe_eval(node.left)
                right = _safe_eval(node.right)
                if isinstance(node.op, ast.Add): return left + right
                elif isinstance(node.op, ast.Sub): return left - right
                elif isinstance(node.op, ast.Mult): return left * right
                elif isinstance(node.op, ast.Div): return left / right
                elif isinstance(node.op, ast.Pow): return left ** right
                raise ValueError(f"Unsupported op: {{type(node.op)}}")
            elif isinstance(node, ast.UnaryOp):
                operand = _safe_eval(node.operand)
                if isinstance(node.op, ast.USub): return -operand
                elif isinstance(node.op, ast.UAdd): return +operand
                raise ValueError(f"Unsupported unary: {{type(node.op)}}")
            raise ValueError(f"Unsupported node: {{type(node)}}")
            
        try:
            tree = ast.parse(formula, mode='eval')
            df[feature_name] = _safe_eval(tree)
        except Exception:
            df[feature_name] = 0
        
    y = df["{exp.parameters.get('target_column', 'target')}"]
    X = df.drop(columns=["{exp.parameters.get('target_column', 'target')}"])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    # Preprocessor
    numeric_features = ["..."] # Automatically detected in runtime
    categorical_features = ["..."]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
        
    # Model
    model_name = "{exp.model_name}"
    params = {exp.parameters.get('best_params', exp.parameters.get('model_params', dict()))}
    
    if model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**params)
    elif model_name == 'LGBMClassifier':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**params)
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**params)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**params)
        
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    pipeline = build_model()
    
    # This assumes numeric/categorical lists are populated
    # In practice, you'd dynamically select them here:
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    pipeline.steps[0][1].transformers[0] = ('num', pipeline.steps[0][1].transformers[0][1], numeric_features)
    pipeline.steps[0][1].transformers[1] = ('cat', pipeline.steps[0][1].transformers[1][1], categorical_features)
    
    pipeline.fit(X_train, y_train)
    print("Training complete! Model Score:", pipeline.score(X_test, y_test))
"""
    return {"script": script, "filename": f"train_{exp.id}.py"}

