"""LocalExperimentExecutor - runs experiments in the local process."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from app.core.errors import step_failed
from ml.core.interfaces import ExperimentRunner
from ml.core.targets import TargetEncoder
from ml.data.preparation.feature_frame import prepare_feature_frame
from ml.experiments.acceptance import compare_feature_sets
from ml.experiments.schema import (
    ExperimentDecision,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from ml.features.safe_eval import InvalidFormula, evaluate, parse
from ml.metrics.classification import compute_classification_metrics
from ml.metrics.importance import IMPORTANCE_METHOD, builtin_importances
from ml.models.engines import DEFAULT_ENGINE, ENGINES, get_engine, one_hot_preprocessor
from ml.validation.splits import SplitPlan, make_splits

# Model engines (ml/models/engines): every model family behind one interface.
MODEL_REGISTRY = ENGINES

# One-hot preprocessing for engines that need numeric input (and the acceptance model).
build_preprocessor = one_hot_preprocessor


ACCEPTANCE_MODEL = {
    "model": "LGBMClassifier",
    "n_estimators": 100,
    "num_leaves": 15,
    "learning_rate": 0.1,
}


def acceptance_pipeline(X: pd.DataFrame) -> Pipeline:
    """Fixed, fast model used only to compare feature sets (same for every candidate)."""
    params = {k: v for k, v in ACCEPTANCE_MODEL.items() if k != "model"}
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", LGBMClassifier(**params, random_state=0, verbose=-1)),
        ]
    )


def predict_once(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict labels (and probabilities when available) with a single model call."""
    if hasattr(pipeline, "predict_proba"):
        prob = pipeline.predict_proba(X)
        return pipeline.classes_[np.argmax(prob, axis=1)], prob
    return pipeline.predict(X), None


class LocalExperimentExecutor(ExperimentRunner):
    """Executes experiments locally (single-process)."""

    def __init__(self, data_loader_func: Callable[[str], pd.DataFrame] | None = None):
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
            # Fails loudly (EngineNotAvailable) for an unknown or uninstalled engine.
            engine = get_engine(spec.model_name or DEFAULT_ENGINE)
            spec.parameters["engine"] = engine.describe()
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
            # Formulas are LLM output: validated and evaluated by the one safe evaluator.
            # An invalid formula rejects the experiment; it never becomes a column of zeros.
            for feat in engineered:
                try:
                    df[feat["name"]] = evaluate(parse(feat["formula"], df.columns), df)
                except InvalidFormula as e:
                    return self._rejected_invalid(spec, feat, e.reason, start_time)

            y = df[target_col]
            X = df.drop(columns=[target_col])

            # In MVP, if feature_set is defined, we want to train on the full original columns PLUS the new feature
            # so we just ensure the new feature exists.
            if spec.feature_set:
                missing_feats = [f for f in spec.feature_set if f not in X.columns]
                if missing_feats:
                    raise ValueError(f"Feature columns not found: {missing_feats}")

            # One split contract (ml/validation/splits.py): tuning and decisions see only
            # train/validation rows; the test rows are scored once at the end.
            plan = SplitPlan.default_for(len(X), spec.validation_config)
            split = make_splits(y, plan)
            random_state = plan.seed
            X_train, X_val, X_test = X.iloc[split.train], X.iloc[split.val], X.iloc[split.test]
            y_train, y_val, y_test = y.iloc[split.train], y.iloc[split.val], y.iloc[split.test]
            # Inner folds as positions within X_train (cv plans only).
            inner_folds = [
                (np.searchsorted(split.train, a), np.searchsorted(split.train, b))
                for a, b in split.folds
            ]
            spec.parameters["split_plan"] = plan.model_dump()
            spec.parameters["split"] = split.summary()
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

            # Each engine brings its own preprocessing (LightGBM: native categories;
            # linear models: one-hot + scaling) unless a preprocessing config overrides it.
            prep_config = spec.preprocessing_config
            custom_preprocessor = None
            if prep_config:
                from ml.data.preparation.dynamic_builder import DynamicPipelineBuilder

                custom_preprocessor = DynamicPipelineBuilder.build(prep_config, target_col)

            def make_pipeline(member, params: dict) -> Pipeline:
                # A fresh preprocessor per pipeline so fits never share state.
                return member.make_pipeline(X_train, params, random_state, custom_preprocessor)

            def validation_predictions(member, params: dict, with_proba: bool = False):
                """Predictions for the validation rows: the holdout split, or out-of-fold
                predictions over the inner folds. Never touches the test rows."""
                if not inner_folds:
                    pipe = make_pipeline(member, params)
                    pipe.fit(X_train, y_train)
                    prob = (
                        pipe.predict_proba(X_val)
                        if with_proba and hasattr(pipe, "predict_proba")
                        else None
                    )
                    return y_val, pipe.predict(X_val), prob
                y_pred = np.empty(len(y_train), dtype=y_train.dtype)
                prob = None
                for fit_idx, val_idx in inner_folds:
                    pipe = make_pipeline(member, params)
                    pipe.fit(X_train.iloc[fit_idx], y_train[fit_idx])
                    y_pred[val_idx] = pipe.predict(X_train.iloc[val_idx])
                    if with_proba and hasattr(pipe, "predict_proba"):
                        fold_prob = pipe.predict_proba(X_train.iloc[val_idx])
                        if prob is None:
                            prob = np.full((len(y_train), fold_prob.shape[1]), np.nan)
                        prob[val_idx] = fold_prob
                return y_train, y_pred, prob

            # 3. Tuning, only over the engine's own search space
            model_params = spec.parameters.get("model_params", {}).copy()
            best_params = model_params.copy()

            try:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                has_optuna = True
            except ImportError:
                has_optuna = False

            # Engines without a search space are not tuned; that would repeat the same fit n_trials times.
            space = engine.search_space()
            if not space:
                spec.parameters["tuning"] = "none: no search space for this model"
            elif not has_optuna:
                spec.parameters["tuning"] = "none: optuna not installed"
            if has_optuna and space:
                n_trials = spec.parameters.get("n_trials", 20)
                spec.parameters["tuning"] = (
                    f"optuna, {n_trials} trials, scored on the validation split"
                    if not inner_folds
                    else f"optuna, {n_trials} trials, scored out-of-fold on {len(inner_folds)} inner folds"
                )

                def objective(trial):
                    params = model_params.copy()
                    for key, (kind, low, high) in space.items():
                        params[key] = (
                            trial.suggest_int(key, low, high)
                            if kind == "int"
                            else trial.suggest_float(key, low, high)
                        )
                    # Tuning never sees the test split: score on validation rows only.
                    y_true_tune, y_pred_tune, _ = validation_predictions(engine, params)
                    return compute_classification_metrics(y_true_tune, y_pred_tune, None)["f1"]

                def run_study():
                    sampler = optuna.samplers.TPESampler(seed=random_state)
                    study = optuna.create_study(direction="maximize", sampler=sampler)
                    study.optimize(objective, n_trials=n_trials)
                    return study.best_params

                try:
                    best_tuned = await asyncio.to_thread(run_study)
                    best_params.update(best_tuned)
                    spec.parameters.setdefault("best_params", {}).update(best_tuned)
                except Exception as e:  # noqa: BLE001 - Optuna or any engine can raise; recorded on the run
                    step_failed(spec.parameters, "tuning", e, skipped="used the given parameters")

            # 4. Validation metrics (holdout, or out-of-fold for small data).
            metric_notes: dict[str, str] = {}
            val_notes: dict[str, str] = {}

            def fit_and_score_val():
                return compute_classification_metrics(
                    *validation_predictions(engine, best_params, with_proba=True), notes=val_notes
                )

            val_metrics = await asyncio.to_thread(fit_and_score_val)

            # Majority-vote ensemble of three built-in engines, reported on the validation split only.
            if spec.parameters.get("ensemble", True):
                try:

                    def ensemble_val_metrics():
                        preds = {}
                        y_true_val = None
                        for name in ("XGBClassifier", "LGBMClassifier", "LogisticRegression"):
                            member = get_engine(name)
                            params = best_params if name == spec.model_name else {}
                            y_true_val, preds[name], _ = validation_predictions(member, params)
                        vote = pd.DataFrame(preds).mode(axis=1)[0].values
                        return compute_classification_metrics(y_true_val, vote, None)

                    for k, v in (await asyncio.to_thread(ensemble_val_metrics)).items():
                        val_metrics[f"ensemble_{k}"] = v
                except Exception as e:  # noqa: BLE001 - any member engine can raise; recorded on the run
                    step_failed(
                        spec.parameters, "ensemble_status", e, skipped="no ensemble metrics"
                    )

            # 5. Final model: refit on train + validation, then touch the test split once.
            X_fit = pd.concat([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])
            pipeline = make_pipeline(engine, best_params)
            await asyncio.to_thread(pipeline.fit, X_fit, y_fit)
            y_pred, y_prob = await asyncio.to_thread(predict_once, pipeline, X_test)
            test_metrics = compute_classification_metrics(
                y_test, y_pred, y_prob, notes=metric_notes
            )

            # Unprefixed keys are the test metrics (what the UI shows); val_* are what
            # tuning and keep/reject decisions may use.
            metrics = dict(test_metrics)
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            metric_notes.update({f"test_{k}": v for k, v in metric_notes.items()})
            metric_notes.update({f"val_{k}": v for k, v in val_notes.items()})
            if metric_notes:
                spec.parameters["metric_notes"] = metric_notes

            # Record the model's own feature importances for the real columns (not SHAP).
            try:
                spec.parameters["feature_importances"] = builtin_importances(
                    pipeline, list(X_train.columns)
                )
                spec.parameters["importance_method"] = IMPORTANCE_METHOD
            except Exception as e:  # noqa: BLE001 - importances are optional; the reason is shown as "not available"
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
                timestamp=datetime.now(UTC),
            )

            self._running[spec.id] = ExperimentStatus.COMPLETED.value
            return result

        except Exception:
            self._running[spec.id] = ExperimentStatus.FAILED.value
            raise

    def _rejected_invalid(
        self, spec: ExperimentSpec, feat: dict, reason: str, start_time: float
    ) -> ExperimentResult:
        """Result for an experiment whose formula failed validation: nothing trained, no metrics."""
        spec.parameters["invalid_formula"] = {
            "name": feat.get("name"),
            "formula": feat.get("formula"),
            "reason": reason,
        }
        self._running[spec.id] = ExperimentStatus.REJECTED_INVALID.value
        return ExperimentResult(
            id=spec.id,
            parent_id=spec.parent_id,
            project_id=spec.project_id,
            dataset_version=spec.dataset_version,
            hypothesis=spec.hypothesis,
            change_description=spec.change_description,
            model_name=spec.model_name,
            parameters=spec.parameters,
            validation_config=spec.validation_config,
            metrics={},
            runtime_seconds=time.time() - start_time,
            cost_usd=0.0,
            status=ExperimentStatus.REJECTED_INVALID,
            decision=ExperimentDecision.REJECT,
            decision_reason=f"Invalid formula for '{feat.get('name')}': {reason}",
            timestamp=datetime.now(UTC),
        )

    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        """Validate an ExperimentSpec. Return list of error strings."""
        errors: list[str] = []
        if not spec.id:
            errors.append("id is required")
        if not spec.project_id:
            errors.append("project_id is required")
        if not spec.hypothesis:
            errors.append("hypothesis is required")
        if not spec.model_name:
            errors.append("model_name is required")
        if not spec.dataset_version:
            errors.append("dataset_version is required")
        if "target_column" not in spec.parameters:
            errors.append("parameters['target_column'] is required")
        return errors

    async def get_status(self, experiment_id: str) -> str:
        return self._running.get(experiment_id, "unknown")

    async def cancel(self, experiment_id: str) -> bool:
        if experiment_id in self._running:
            self._running[experiment_id] = ExperimentStatus.FAILED.value
            return True
        return False
