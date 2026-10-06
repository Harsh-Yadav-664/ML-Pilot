"""UI Adapter API endpoints for the Frontend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ai.gateway import AIGateway
from app.api.deps import DBSession, Gateway
from app.core.config import settings
from app.core.datasets import DATASETS_DIR, UPLOAD_DIR, safe_dataset_path
from app.schemas.experiment import ExperimentCreate, ExperimentRead
from app.services.experiment_service import ExperimentService, champion_path
from ml.agents.decision_agent import DecisionAgent
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.ingestion.sql_loader import SqlLoader, redact
from ml.data.preparation.feature_frame import ONEHOT_MAX_CATEGORIES
from ml.data.preparation.native.native_prep import NativeDataPreparationProvider
from ml.data.profiling.profiler import DataProfiler
from ml.features import safe_eval as safe_eval_module
from ml.models.engines import DEFAULT_ENGINE, EngineNotAvailable, get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["ui-adapter"])

# For MVP, default if not provided
DEMO_PROJECT_ID = "demo-project-id"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

from pydantic import BaseModel


def _required(data: dict[str, Any], key: str) -> Any:
    """Return a required request field, or fail with HTTP 400 instead of guessing a default."""
    value = data.get(key)
    if not value:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return value


class SqlConnectRequest(BaseModel):
    connection_string: str
    query: str


@router.post("/data/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV dataset and return its columns."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_id = str(uuid.uuid4())
    # Keep only the base name so a crafted filename can't escape the upload dir
    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

    def save_upload() -> None:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    await asyncio.to_thread(save_upload)

    try:
        loader = CsvLoader()
        df = loader.load(file_path)
        columns = df.columns.tolist()
        return {
            "dataset_path": safe_dataset_path(file_path),
            "filename": safe_name,
            "columns": columns,
            "total_rows": len(df),
        }
    except Exception as e:  # noqa: BLE001 - any parse error becomes an HTTP 400
        logger.error(f"Failed to parse uploaded CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


class SampleDataRequest(BaseModel):
    dataset_name: str = "telecom_churn"


@router.post("/data/sample")
async def load_sample_dataset(request: SampleDataRequest) -> dict[str, Any]:
    """Load a pre-bundled sample dataset."""
    filename = f"{request.dataset_name}.csv"
    source_path = os.path.join(DATASETS_DIR, filename)

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found")

    try:
        loader = CsvLoader()
        df = loader.load(source_path)
        columns = df.columns.tolist()

        # Pre-select a likely target for the UI to be helpful
        default_target = "Churn" if "Churn" in columns else (columns[-1] if columns else "")

        return {
            "dataset_path": safe_dataset_path(source_path),
            "filename": filename,
            "columns": columns,
            "total_rows": len(df),
            "default_target": default_target,
        }
    except Exception as e:  # noqa: BLE001 - any load error becomes an HTTP 500
        logger.error(f"Failed to load sample data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load sample data: {e!s}")


@router.post("/data/connect-sql")
async def connect_sql(request: SqlConnectRequest) -> dict[str, Any]:
    """Connect to a SQL database and extract a dataset."""
    try:
        loader = SqlLoader()
        df = loader.load(request.connection_string, request.query)
        columns = df.columns.tolist()

        # Save snapshot to disk so the rest of the file-based pipeline works identically
        file_id = loader.hash_connection(request.connection_string, request.query)
        file_path = os.path.join(UPLOAD_DIR, f"sql_{file_id}.csv")
        df.to_csv(file_path, index=False)

        return {
            "dataset_path": file_path,
            "filename": f"SQL Query Snapshot ({file_id})",
            "columns": columns,
            "total_rows": len(df),
        }
    except Exception as e:  # noqa: BLE001 - any driver error becomes a redacted HTTP 400
        # Never echo the connection string or password back or into logs.
        message = redact(str(e), request.connection_string)
        logger.error(f"Failed to connect and query SQL: {message}")
        raise HTTPException(
            status_code=400, detail=f"Failed to connect and query SQL: {message}"
        ) from None


@router.get("/data/metrics")
async def get_data_metrics(dataset_path: str, target_column: str = "target") -> dict[str, Any]:
    """Return dataset metrics for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(safe_dataset_path(dataset_path))
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_column)

        return {
            "total_rows": profile.rows,
            "total_columns": profile.columns,
            "missing_data_percent": float(profile.missing_rate),
            "duplicate_rows": profile.duplicate_rows,
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=400, detail=f"Could not profile {dataset_path}: {e}") from e


def _column_profile(
    series: pd.Series, name: str, target_column: str, n_rows: int
) -> dict[str, Any]:
    """Real per-column summary for the UI: dtype, missing %, unique count, 12-bin shape."""
    non_null = series.dropna()
    unique = int(non_null.nunique())
    if pd.api.types.is_bool_dtype(series):
        dtype = "bool"
    elif pd.api.types.is_integer_dtype(series):
        dtype = "int"
    elif pd.api.types.is_float_dtype(series):
        dtype = "float"
    elif pd.api.types.is_datetime64_any_dtype(series):
        dtype = "datetime"
    elif unique <= 50:
        dtype = "category"
    else:
        dtype = "string"

    if dtype in ("int", "float") and unique > 12:
        counts, _ = np.histogram(non_null.astype(float), bins=12)
        dist = counts.tolist()
    else:
        dist = non_null.astype(str).value_counts().head(12).tolist()
    peak = max(dist) if dist else 0
    dist = [round(c / peak, 4) if peak else 0.0 for c in dist] + [0.0] * (12 - len(dist))

    if name == target_column:
        role = "target"
    elif dtype in ("string", "int") and n_rows > 0 and unique == n_rows:
        role = "id"
    else:
        role = "feature"
    return {
        "name": name,
        "dtype": dtype,
        "role": role,
        "missing_pct": round(float(series.isna().mean()) * 100, 2),
        "unique": unique,
        "dist": dist,
    }


@router.get("/data/columns")
async def get_data_columns(dataset_path: str, target_column: str) -> list[dict[str, Any]]:
    """Return a real profile of every column for the UI."""
    try:
        df = CsvLoader().load(safe_dataset_path(dataset_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load {dataset_path}: {e}") from e
    return [_column_profile(df[c], c, target_column, len(df)) for c in df.columns]


# Engines the generated training script can rebuild.
EXPORTABLE_ENGINES = {
    "XGBClassifier",
    "LGBMClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "LogisticRegression",
}


@router.get("/data/leakage-warnings")
async def get_leakage_warnings(
    dataset_path: str, target_column: str = "target"
) -> list[dict[str, Any]]:
    """Return leakage warnings for the UI."""
    try:
        loader = CsvLoader()
        df = loader.load(safe_dataset_path(dataset_path))
        prep = NativeDataPreparationProvider()
        findings = prep.detect_leakage(df, target_column=target_column)
        # The UI's severity scale: block -> high, warn -> medium, info -> low.
        ui_severity = {"block": "high", "warn": "medium", "info": "low"}
        return [
            {
                "id": f"warn_{i}",
                "column": f.column or "(rows)",
                "message": f.explanation,
                "severity": ui_severity[f.severity],
                "category": f.category,
                "check": f.check,
                "evidence": f.evidence,
            }
            for i, f in enumerate(findings)
        ]
    except Exception as e:
        logger.error(f"Failed to get leakage: {e}")
        raise HTTPException(status_code=400, detail=f"Leakage scan failed: {e}") from e


@router.get("/agent/suggestions")
async def get_agent_suggestions(
    db: DBSession,
    gateway: Gateway,
    dataset_path: str,
    target_column: str = "target",
    user_query: str | None = None,
) -> list[dict[str, Any]]:
    """Trigger AI and return suggestions."""
    svc = ExperimentService(db)

    objective = "Maximize F1 score while preventing overfitting"
    if user_query:
        objective += (
            f". User strictly requested: '{user_query}'. Generate features reflecting this."
        )
    dataset_path = safe_dataset_path(dataset_path)

    try:
        hypotheses = await svc.suggest_experiments(
            dataset_version=dataset_path,
            target_column=target_column,
            objective=objective,
            max_hypotheses=3,
            gateway=gateway,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=502, detail=f"AI suggestions failed: {e}") from e
    return hypotheses


async def background_runner(experiment_id: str):
    """Thin wrapper: ExperimentService.run_experiment_background opens its own session."""
    try:
        svc = ExperimentService(None)  # session unused — service opens its own
        await svc.run_experiment_background(experiment_id)
    except Exception:  # last-resort guard; the service already marks the experiment failed
        logger.exception(f"Background task failed for experiment {experiment_id}")


@router.post("/experiments/run")
async def run_experiment(
    data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks
) -> ExperimentRead:
    """Run an experiment based on a suggestion."""
    svc = ExperimentService(db)
    suggestion = data.get("feature_suggestion", {})
    dataset_path = safe_dataset_path(_required(data, "dataset_path"))
    target_column = data.get("target_column", "target")

    model_name = data.get("model_name") or DEFAULT_ENGINE
    try:
        get_engine(model_name)
    except EngineNotAvailable as e:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {e}") from e

    exp_create = ExperimentCreate(
        project_id=DEMO_PROJECT_ID,
        parent_id=data.get("parent_id"),
        dataset_version=dataset_path,
        hypothesis=suggestion.get("reason", "No reason provided"),
        change_description=f"Added feature: {suggestion.get('name')} via formula {suggestion.get('formula')}",
        model_name=model_name,
        feature_set=[suggestion.get("name")] if suggestion.get("name") else [],
        parameters={
            "target_column": target_column,
            "feature_name": suggestion.get("name"),
            "formula": suggestion.get("formula"),
        },
    )

    exp = await svc.create(exp_create)
    # Commit before scheduling: the background run reads the row from its own session.
    await db.commit()
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)


@router.post("/experiments/baseline")
async def run_baseline(
    data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks
) -> ExperimentRead:
    """Run a deterministic baseline experiment."""
    svc = ExperimentService(db)
    dataset_path = safe_dataset_path(_required(data, "dataset_path"))
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
            "model_params": {"n_estimators": 50, "random_state": 42},
        },
    )

    exp = await svc.create(exp_create)
    # Commit before scheduling: the background run reads the row from its own session.
    await db.commit()
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)


@router.get("/experiments/tree")
async def get_experiment_tree(
    db: DBSession, dataset_path: str | None = None
) -> list[dict[str, Any]]:
    """Return experiments as a flat list, optionally only those on one dataset."""
    svc = ExperimentService(db)
    exps, _ = await svc.list_by_project(DEMO_PROJECT_ID, page=1, page_size=500)
    if dataset_path:
        wanted = safe_dataset_path(dataset_path)
        exps = [e for e in exps if e.dataset_version == wanted]

    path_ids = [e.id for e in champion_path(exps)]
    result = []
    for e in exps:
        params = e.parameters or {}
        result.append(
            {
                "id": e.id,
                "parent_id": e.parent_id,
                "model_name": e.model_name,
                "status": e.status,
                "metrics": e.metrics or {},
                "runtime_seconds": e.runtime_seconds or 0.0,
                "created_at": e.created_at.isoformat(),
                "title": e.change_description,
                "feature": params.get("feature_name"),
                "decision": e.decision
                if e.decision in ("keep", "reject")
                else ("baseline" if not e.parent_id else "none"),
                "error": e.decision_reason if e.status in ("failed", "rejected_invalid") else None,
                # 'fallback' when the idea came from the offline stub instead of a real LLM.
                "hypothesis_mode": (params.get("hypothesis_llm") or {}).get("decision_mode"),
                "on_champion_path": e.id in path_ids,
                "champion": bool(path_ids) and e.id == path_ids[-1],
            }
        )
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
    except Exception as e:  # recorded on the job as status 'failed' with the message
        logger.exception("Auto-optimize failed")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@router.post("/agent/auto-optimize")
async def auto_optimize(
    data: dict[str, Any], db: DBSession, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Trigger autonomous optimization loop."""
    dataset_path = safe_dataset_path(_required(data, "dataset_path"))
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
    background_tasks: BackgroundTasks,
    gateway: Gateway,
) -> ExperimentRead:
    """Uses AI to generate an advanced cleaning strategy and runs it as a baseline."""
    dataset_path = safe_dataset_path(_required(data, "dataset_path"))
    target_column = data.get("target_column", "target")

    # 1. Profile Data
    loader = CsvLoader()
    df = loader.load(dataset_path)
    profiler = DataProfiler()
    profile = profiler.profile(df, target_column=target_column)

    # 2. Get AI Strategy
    from ml.agents.cleaning_agent import CleaningStrategyError, DataCleaningAgent

    agent = DataCleaningAgent(gateway)
    try:
        prep_config = await agent.generate_cleaning_strategy(profile, target_column)
    except CleaningStrategyError as e:
        raise HTTPException(status_code=502, detail=f"AI cleaning strategy failed: {e}") from e

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
            "model_params": {"n_estimators": 50, "random_state": 42},
        },
    )

    exp = await svc.create(exp_create)
    # Commit before scheduling: the background run reads the row from its own session.
    await db.commit()
    background_tasks.add_task(background_runner, exp.id)
    return ExperimentRead.model_validate(exp)


@router.get("/experiments/{experiment_id}/export")
async def export_experiment_script(
    experiment_id: str, db: DBSession, dataset_path: str | None = None
):
    """Export a python training script for an experiment, or for "champion" of a dataset."""
    svc = ExperimentService(db)
    if experiment_id == "champion":
        if not dataset_path:
            raise HTTPException(
                status_code=400, detail="dataset_path is required to export the champion"
            )
        wanted = safe_dataset_path(dataset_path)
        exps, _ = await svc.list_by_project(DEMO_PROJECT_ID, page=1, page_size=500)
        path = champion_path([e for e in exps if e.dataset_version == wanted])
        if not path:
            raise HTTPException(
                status_code=404, detail="No completed baseline for this dataset yet"
            )
        experiment_id = path[-1].id
    exp = await svc.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if exp.model_name not in EXPORTABLE_ENGINES:
        raise HTTPException(
            status_code=409,
            detail=f"Export for the {exp.model_name} engine is not available yet; supported: {sorted(EXPORTABLE_ENGINES)}.",
        )
    target_encoding = exp.parameters.get("target_encoding")
    if not target_encoding:
        raise HTTPException(
            status_code=409,
            detail="Experiment has no recorded target encoding; run it to completion before exporting.",
        )
    classes_literal = json.dumps(target_encoding["classes"])
    engineered = list(exp.parameters.get("features") or [])
    if exp.parameters.get("feature_name") and exp.parameters.get("formula"):
        engineered.append(
            {"name": exp.parameters["feature_name"], "formula": exp.parameters["formula"]}
        )
    features_literal = json.dumps(engineered)
    excluded_literal = json.dumps(sorted(exp.parameters.get("excluded_features", {})))
    numeric_text_literal = json.dumps(sorted(exp.parameters.get("numeric_coercion", {})))
    # The script embeds the same safe evaluator the experiment used (no second copy to drift).
    safe_eval_source = Path(safe_eval_module.__file__).read_text()

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

import json

# --- Safe formula evaluator, copied from MLPilot (ml/features/safe_eval.py) ---
{safe_eval_source}
# --- end of safe formula evaluator ---

# Engineered features used by this experiment, in the order they were added
FEATURES = json.loads({features_literal!r})

# ID columns excluded from features, and text columns converted to numbers, when the experiment ran
EXCLUDED_FEATURES = json.loads({excluded_literal!r})
NUMERIC_TEXT_COLUMNS = json.loads({numeric_text_literal!r})

def load_and_prepare_data():
    df = pd.read_csv("{exp.dataset_version}")

    # Same clean-up MLPilot applied when the experiment ran
    df = df.drop(columns=EXCLUDED_FEATURES)
    for col in NUMERIC_TEXT_COLUMNS:
        df[col] = pd.to_numeric(df[col].astype("string").str.strip(), errors="coerce").astype("float64")

    # Feature engineering with the same safe evaluator MLPilot used; an invalid formula raises.
    for feat in FEATURES:
        df[feat["name"]] = evaluate_formula(feat["formula"], df)

    y = df["{exp.parameters.get("target_column", "target")}"]
    X = df.drop(columns=["{exp.parameters.get("target_column", "target")}"])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Class labels in encoded order (index = model output); recorded when the experiment ran
CLASSES = json.loads({classes_literal!r})
CLASS_INDEX = {{label: i for i, label in enumerate(CLASSES)}}

def encode_target(y):
    return y.map(CLASS_INDEX).astype(int)

def decode_predictions(codes):
    return [CLASSES[int(c)] for c in codes]

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
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories={ONEHOT_MAX_CATEGORIES}, sparse_output=False))
            ]), categorical_features)
        ])
        
    # Model
    model_name = "{exp.model_name}"
    params = {exp.parameters.get("best_params", exp.parameters.get("model_params", {}))}
    
    if model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**params)
    elif model_name == 'LGBMClassifier':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**params)
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**params)
    elif model_name == 'GradientBoostingClassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**params)
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
    
    pipeline.fit(X_train, encode_target(y_train))
    print("Training complete! Accuracy:", pipeline.score(X_test, encode_target(y_test)))
    predictions = decode_predictions(pipeline.predict(X_test))
    print("Sample predictions:", predictions[:10])
"""
    return {"script": script, "filename": f"train_{exp.id}.py"}
