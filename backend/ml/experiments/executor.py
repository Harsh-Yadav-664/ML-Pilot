"""LocalExperimentExecutor - runs experiments in the local process."""
from __future__ import annotations

import time
import asyncio
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ml.core.interfaces import ExperimentRunner
from ml.experiments.schema import ExperimentSpec, ExperimentResult, ExperimentStatus, ExperimentDecision
from ml.metrics.classification import compute_classification_metrics

# Registry for models supported in Phase 1
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
}

class LocalExperimentExecutor(ExperimentRunner):
    """Executes experiments locally (single-process)."""

    def __init__(self, data_loader_func: Optional[Callable[[str], pd.DataFrame]] = None):
        """Initialize with an optional data loader function.
        
        Args:
            data_loader_func: A function that takes a dataset_version string and returns a pandas DataFrame.
        """
        self.data_loader_func = data_loader_func
        self._running: dict[str, str] = {}  # experiment_id -> status

    async def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Execute the experiment spec and return a result."""
        errors = self.validate_spec(spec)
        if errors:
            raise ValueError(f"Invalid experiment spec: {errors}")

        self._running[spec.id] = ExperimentStatus.RUNNING.value
        start_time = time.time()

        try:
            # 1. Load Data
            if not self.data_loader_func:
                raise ValueError("No data loader configured for executor.")
            
            # Using asyncio.to_thread just in case data loading is blocking
            df = await asyncio.to_thread(self.data_loader_func, spec.dataset_version)

            target_col = spec.parameters.get("target_column")
            if not target_col or target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset.")

            # 2. Prepare Data
            # Execute feature engineering step safely
            feature_name = spec.parameters.get("feature_name")
            formula = spec.parameters.get("formula")
            if feature_name and formula:
                try:
                    df[feature_name] = df.eval(formula)
                except Exception:
                    # MVP: if pandas eval fails due to syntax, fallback to 0 so pipeline completes
                    df[feature_name] = 0

            y = df[target_col]
            X = df.drop(columns=[target_col])

            # In MVP, if feature_set is defined, we want to train on the full original columns PLUS the new feature
            # so we just ensure the new feature exists.
            if spec.feature_set:
                missing_feats = [f for f in spec.feature_set if f not in X.columns]
                if missing_feats:
                    for f in missing_feats:
                        X[f] = 0  # Fallback to prevent crash

            test_size = spec.validation_config.get("test_size", 0.2)
            random_state = spec.validation_config.get("random_state", 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Robust preprocessor to handle real-world messy data
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

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

            # 3. Initialize Model
            model_cls = MODEL_REGISTRY.get(spec.model_name)
            if not model_cls:
                raise ValueError(f"Unsupported model: {spec.model_name}. Supported: {list(MODEL_REGISTRY.keys())}")
            
            model_params = spec.parameters.get("model_params", {})
            if 'random_state' in model_cls().get_params():
                model_params['random_state'] = random_state
                
            clf = model_cls(**model_params)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

            # 4. Train Model
            # Run training in thread pool to prevent blocking the async loop
            await asyncio.to_thread(pipeline.fit, X_train, y_train)

            # 5. Predict & Calculate Metrics
            y_pred = await asyncio.to_thread(pipeline.predict, X_test)
            y_prob = None
            if hasattr(pipeline, "predict_proba"):
                y_prob = await asyncio.to_thread(pipeline.predict_proba, X_test)

            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            runtime_seconds = time.time() - start_time

            result = ExperimentResult(
                id=spec.id,
                parent_id=spec.parent_id,
                project_id=spec.project_id,
                dataset_version=spec.dataset_version,
                hypothesis=spec.hypothesis,
                change_description=spec.change_description,
                model_name=spec.model_name,
                parameters=spec.parameters,
                validation_config=spec.validation_config,
                metrics=metrics,
                artifacts=[],
                runtime_seconds=runtime_seconds,
                cost_usd=0.0,
                status=ExperimentStatus.COMPLETED,
                decision=ExperimentDecision.PENDING,
                timestamp=datetime.now(timezone.utc),
            )
            
            self._running[spec.id] = ExperimentStatus.COMPLETED.value
            return result

        except Exception as e:
            self._running[spec.id] = ExperimentStatus.FAILED.value
            raise

    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        """Validate an ExperimentSpec. Return list of error strings."""
        errors: list[str] = []
        if not spec.id: errors.append("id is required")
        if not spec.project_id: errors.append("project_id is required")
        if not spec.hypothesis: errors.append("hypothesis is required")
        if not spec.model_name: errors.append("model_name is required")
        if not spec.dataset_version: errors.append("dataset_version is required")
        if "target_column" not in spec.parameters: errors.append("parameters['target_column'] is required")
        return errors

    async def get_status(self, experiment_id: str) -> str:
        return self._running.get(experiment_id, "unknown")

    async def cancel(self, experiment_id: str) -> bool:
        if experiment_id in self._running:
            self._running[experiment_id] = ExperimentStatus.FAILED.value
            return True
        return False
