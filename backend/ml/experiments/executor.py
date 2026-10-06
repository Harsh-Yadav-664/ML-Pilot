"""LocalExperimentExecutor - runs experiments in the local process."""
from __future__ import annotations

import time
import asyncio
import numpy as np
import pandas as pd
import ast
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sklearn.base import clone
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
from ml.core.targets import TargetEncoder
from ml.data.preparation.feature_frame import ONEHOT_MAX_CATEGORIES, prepare_feature_frame
from ml.experiments.schema import ExperimentSpec, ExperimentResult, ExperimentStatus, ExperimentDecision
from ml.experiments.acceptance import compare_feature_sets
from ml.metrics.classification import compute_classification_metrics
from ml.metrics.importance import IMPORTANCE_METHOD, builtin_importances

# Registry for models supported in Phase 1
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
}

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute and scale numeric columns; impute and one-hot encode the rest (capped)."""
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.columns.difference(numeric_features, sort=False)
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=ONEHOT_MAX_CATEGORIES, sparse_output=False))
            ]), categorical_features)
        ])


ACCEPTANCE_MODEL = {"model": "LGBMClassifier", "n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1}


def acceptance_pipeline(X: pd.DataFrame) -> Pipeline:
    """Fixed, fast model used only to compare feature sets (same for every candidate)."""
    params = {k: v for k, v in ACCEPTANCE_MODEL.items() if k != "model"}
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor(X)),
        ('classifier', LGBMClassifier(**params, random_state=0, verbose=-1)),
    ])


def split_train_val_test(
    X: pd.DataFrame, y: pd.Series, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    """Split rows into train / validation / test with a recorded, seeded definition.

    Tuning and keep/reject decisions use validation; the test split is scored once.
    Stratified by the target when every class has enough rows.
    """
    test_size = config.get("test_size", 0.2)
    val_size = config.get("val_size", 0.2)
    random_state = config.get("random_state", 42)
    stratified = bool(y.value_counts().min() >= 5)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratified else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), random_state=random_state,
        stratify=y_rest if stratified else None,
    )
    info = {
        "method": "random holdout",
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "stratified": stratified,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, info


def predict_once(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict labels (and probabilities when available) with a single model call."""
    if hasattr(pipeline, "predict_proba"):
        prob = pipeline.predict_proba(X)
        return pipeline.classes_[np.argmax(prob, axis=1)], prob
    return pipeline.predict(X), None


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
            # Drop ID columns and convert numbers stored as text; record both on the run.
            features, frame_report = prepare_feature_frame(df.drop(columns=[target_col]))
            spec.parameters.update(frame_report)
            df = features.assign(**{target_col: df[target_col]})

            # Execute feature engineering step safely
            # Engineered features: the champion's accepted ones ("features"), then the
            # candidate being tested ("feature_name"/"formula"), evaluated in order.
            feature_name = spec.parameters.get("feature_name")
            formula = spec.parameters.get("formula")
            engineered = [dict(f) for f in spec.parameters.get("features") or []]
            candidate = feature_name if feature_name and formula else None
            if candidate:
                engineered.append({"name": feature_name, "formula": formula})
            if engineered:
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

                for feat in engineered:
                    try:
                        tree = ast.parse(feat["formula"], mode='eval')
                        df[feat["name"]] = _safe_eval(tree)
                    except Exception:
                        # MVP: if safe eval fails due to syntax or unsupported node, fallback to 0
                        # (a constant column is then rejected by the acceptance rule; see [1.4]).
                        df[feat["name"]] = 0

            y = df[target_col]
            X = df.drop(columns=[target_col])

            # In MVP, if feature_set is defined, we want to train on the full original columns PLUS the new feature
            # so we just ensure the new feature exists.
            if spec.feature_set:
                missing_feats = [f for f in spec.feature_set if f not in X.columns]
                if missing_feats:
                    for f in missing_feats:
                        X[f] = 0  # Fallback to prevent crash

            random_state = spec.validation_config.get("random_state", 42)
            X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_train_val_test(
                X, y, spec.validation_config
            )
            spec.parameters["split"] = split_info
            spec.parameters["feature_columns"] = list(X_train.columns)

            # Encode class labels (fit on training labels only) and record the
            # mapping so predictions and exports can be decoded.
            target_encoder = TargetEncoder.fit(
                y_train, positive_class=spec.parameters.get("positive_class")
            )
            y_train = target_encoder.transform(y_train)
            y_val = target_encoder.transform(y_val)
            y_test = target_encoder.transform(y_test)
            spec.parameters["target_encoding"] = target_encoder.to_dict()

            # Keep/reject is decided here, in code: does the candidate beat the noise of a
            # paired repeated-CV comparison on training rows (train + validation, never test)?
            if candidate:
                X_fit_rows = pd.concat([X_train, X_val])
                y_fit_rows = np.concatenate([y_train, y_val])
                gain = await asyncio.to_thread(
                    compare_feature_sets,
                    X_fit_rows.drop(columns=[candidate]),
                    X_fit_rows,
                    y_fit_rows,
                    acceptance_pipeline,
                    spec.parameters.get("acceptance_rule"),
                    random_state,
                )
                spec.parameters["acceptance"] = {**gain.to_dict(), "model": ACCEPTANCE_MODEL}

            # Robust preprocessor to handle real-world messy data
            prep_config = spec.preprocessing_config
            if prep_config:
                from ml.data.preparation.dynamic_builder import DynamicPipelineBuilder
                preprocessor = DynamicPipelineBuilder.build(prep_config, target_col)
            else:
                preprocessor = build_preprocessor(X_train)

            def make_pipeline(estimator) -> Pipeline:
                # Each pipeline gets its own preprocessor so fits never share state.
                return Pipeline(steps=[('preprocessor', clone(preprocessor)), ('classifier', estimator)])

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

            # Only XGBoost and LightGBM have a search space; tuning anything else would
            # repeat the same fit n_trials times.
            tunable = spec.model_name in ('XGBClassifier', 'LGBMClassifier')
            if not tunable:
                spec.parameters['tuning'] = 'none: no search space for this model'
            elif not has_optuna:
                spec.parameters['tuning'] = 'none: optuna not installed'
            if has_optuna and tunable:
                n_trials = spec.parameters.get('n_trials', 20)
                spec.parameters['tuning'] = f'optuna, {n_trials} trials, scored on the validation split'

                def objective(trial):
                    params = model_params.copy()
                    params.update({
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    })
                    if 'random_state' in model_cls().get_params():
                        params['random_state'] = random_state
                    # Tuning never sees the test split: fit on train, score on validation.
                    pipe_tune = make_pipeline(model_cls(**params))
                    pipe_tune.fit(X_train, y_train)
                    y_pred_tune = pipe_tune.predict(X_val)
                    return compute_classification_metrics(y_val, y_pred_tune, None)['f1']

                def run_study():
                    sampler = optuna.samplers.TPESampler(seed=random_state)
                    study = optuna.create_study(direction='maximize', sampler=sampler)
                    study.optimize(objective, n_trials=n_trials)
                    return study.best_params

                try:
                    best_tuned = await asyncio.to_thread(run_study)
                    best_params.update(best_tuned)
                    spec.parameters.setdefault('best_params', {}).update(best_tuned)
                except Exception as e:
                    spec.parameters['tuning'] = f'failed, using the given parameters: {e}'

            if 'random_state' in model_cls().get_params():
                best_params['random_state'] = random_state

            # 4. Validation metrics: fit on train, score on validation.
            def fit_and_score_val():
                pipe = make_pipeline(model_cls(**best_params))
                pipe.fit(X_train, y_train)
                prob = pipe.predict_proba(X_val) if hasattr(pipe, "predict_proba") else None
                return compute_classification_metrics(y_val, pipe.predict(X_val), prob)

            val_metrics = await asyncio.to_thread(fit_and_score_val)

            # Majority-vote ensemble, reported on the validation split only.
            if spec.parameters.get('ensemble', True):
                try:
                    def ensemble_val_metrics():
                        members = [
                            XGBClassifier(**(best_params if spec.model_name == 'XGBClassifier' else {'random_state': random_state})),
                            LGBMClassifier(**(best_params if spec.model_name == 'LGBMClassifier' else {'random_state': random_state, 'verbose': -1})),
                            LogisticRegression(**(best_params if spec.model_name == 'LogisticRegression' else {'random_state': random_state, 'max_iter': 1000})),
                        ]
                        preds = {}
                        for i, member in enumerate(members):
                            pipe = make_pipeline(member)
                            pipe.fit(X_train, y_train)
                            preds[f'p{i}'] = pipe.predict(X_val)
                        vote = pd.DataFrame(preds).mode(axis=1)[0].values
                        return compute_classification_metrics(y_val, vote, None)

                    for k, v in (await asyncio.to_thread(ensemble_val_metrics)).items():
                        val_metrics[f'ensemble_{k}'] = v
                except Exception as e:
                    spec.parameters['ensemble_error'] = str(e)

            # 5. Final model: refit on train + validation, then touch the test split once.
            X_fit = pd.concat([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])
            pipeline = make_pipeline(model_cls(**best_params))
            await asyncio.to_thread(pipeline.fit, X_fit, y_fit)
            y_pred, y_prob = await asyncio.to_thread(predict_once, pipeline, X_test)
            test_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

            # Unprefixed keys are the test metrics (what the UI shows); val_* are what
            # tuning and keep/reject decisions may use.
            metrics = dict(test_metrics)
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            # Record the model's own feature importances for the real columns (not SHAP).
            try:
                spec.parameters["feature_importances"] = builtin_importances(pipeline, list(X_train.columns))
                spec.parameters["importance_method"] = IMPORTANCE_METHOD
            except Exception as e:
                spec.parameters["feature_importances"] = {}
                spec.parameters["importance_method"] = f"not available: {e}"

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
