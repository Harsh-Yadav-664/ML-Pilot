"""LocalExperimentExecutor - runs experiments in the local process."""
from __future__ import annotations

import time
import asyncio
import pandas as pd
import ast
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
                def _safe_eval(node):
                    if isinstance(node, ast.Expression):
                        return _safe_eval(node.body)
                    elif isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.Name):
                        if node.id in df.columns:
                            return df[node.id]
                        raise ValueError(f"Column '{node.id}' not found")
                    elif isinstance(node, ast.BinOp):
                        left = _safe_eval(node.left)
                        right = _safe_eval(node.right)
                        if isinstance(node.op, ast.Add): return left + right
                        elif isinstance(node.op, ast.Sub): return left - right
                        elif isinstance(node.op, ast.Mult): return left * right
                        elif isinstance(node.op, ast.Div): return left / right
                        elif isinstance(node.op, ast.Pow): return left ** right
                        raise ValueError(f"Unsupported op: {type(node.op)}")
                    elif isinstance(node, ast.UnaryOp):
                        operand = _safe_eval(node.operand)
                        if isinstance(node.op, ast.USub): return -operand
                        elif isinstance(node.op, ast.UAdd): return +operand
                        raise ValueError(f"Unsupported unary: {type(node.op)}")
                    raise ValueError(f"Unsupported node: {type(node)}")

                try:
                    tree = ast.parse(formula, mode='eval')
                    df[feature_name] = _safe_eval(tree)
                except Exception:
                    # MVP: if safe eval fails due to syntax or unsupported node, fallback to 0
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
            prep_config = spec.preprocessing_config
            if prep_config:
                from ml.data.preparation.dynamic_builder import DynamicPipelineBuilder
                preprocessor = DynamicPipelineBuilder.build(prep_config, target_col)
            else:
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

            # 3. Initialize Model and Optuna Tuning
            model_cls = MODEL_REGISTRY.get(spec.model_name)
            if not model_cls:
                raise ValueError(f"Unsupported model: {spec.model_name}. Supported: {list(MODEL_REGISTRY.keys())}")
            
            model_params = spec.parameters.get("model_params", {}).copy()
            best_params = model_params.copy()
            
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                has_optuna = True
            except ImportError:
                has_optuna = False

            if has_optuna:
                n_trials = spec.parameters.get('n_trials', 20)

                def objective(trial):
                    params = model_params.copy()
                    if spec.model_name == 'XGBClassifier':
                        params.update({
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        })
                    elif spec.model_name == 'LGBMClassifier':
                        params.update({
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        })
                    else:
                        pass
                    
                    if 'random_state' in model_cls().get_params():
                        params['random_state'] = random_state
                        
                    clf_tune = model_cls(**params)
                    pipe_tune = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_tune)])
                    pipe_tune.fit(X_train, y_train)
                    y_pred_tune = pipe_tune.predict(X_test)
                    metrics_tune = compute_classification_metrics(y_test, y_pred_tune, None)
                    return metrics_tune.get('f1', 0.0)

                def run_study():
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials)
                    return study.best_params

                try:
                    best_tuned = await asyncio.to_thread(run_study)
                    best_params.update(best_tuned)
                    if 'best_params' not in spec.parameters:
                        spec.parameters['best_params'] = {}
                    spec.parameters['best_params'].update(best_tuned)
                except Exception as e:
                    pass

            if 'random_state' in model_cls().get_params():
                best_params['random_state'] = random_state
                
            clf = model_cls(**best_params)
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

            # Auto-Ensembling
            if spec.parameters.get('ensemble', True):
                try:
                    xgb_params = best_params if spec.model_name == 'XGBClassifier' else {}
                    lgb_params = best_params if spec.model_name == 'LGBMClassifier' else {}
                    lr_params = best_params if spec.model_name == 'LogisticRegression' else {}
                    
                    if 'random_state' in XGBClassifier().get_params(): xgb_params['random_state'] = random_state
                    if 'random_state' in LGBMClassifier().get_params(): lgb_params['random_state'] = random_state
                    if 'random_state' in LogisticRegression().get_params(): lr_params['random_state'] = random_state

                    def train_ensemble():
                        m1 = XGBClassifier(**xgb_params)
                        m2 = LGBMClassifier(**lgb_params)
                        m3 = LogisticRegression(**lr_params)
                        p1 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', m1)])
                        p2 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', m2)])
                        p3 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', m3)])
                        p1.fit(X_train, y_train)
                        p2.fit(X_train, y_train)
                        p3.fit(X_train, y_train)
                        return p1, p2, p3

                    p1, p2, p3 = await asyncio.to_thread(train_ensemble)

                    def predict_ensemble():
                        pred1 = p1.predict(X_test)
                        pred2 = p2.predict(X_test)
                        pred3 = p3.predict(X_test)
                        stacked = pd.DataFrame({'p1': pred1, 'p2': pred2, 'p3': pred3})
                        return stacked.mode(axis=1)[0].values

                    y_pred_ens = await asyncio.to_thread(predict_ensemble)
                    ensemble_metrics = compute_classification_metrics(y_test, y_pred_ens, None)
                    for k, v in ensemble_metrics.items():
                        metrics[f'ensemble_{k}'] = v
                except Exception as e:
                    pass

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
