"""
DecisionAgent: Autonomously orchestrates the full hypothesis -> experiment -> decision loop.
Given a dataset, it will:
  1. Profile the dataset
  2. Generate N hypotheses via the AI Gateway
  3. Run all experiments in parallel
  4. Evaluate results and mark each as KEEP/REJECT using LLM reasoning
  5. Return a summary with the winning feature set
"""
from __future__ import annotations

import logging
from typing import Any

from ai.gateway import AIGateway
from ai.router import TaskType
from app.db.session import AsyncSessionLocal
from app.services.experiment_service import ExperimentService
from app.schemas.experiment import ExperimentCreate
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.profiling.profiler import DataProfiler
from ml.experiments.planner import ExperimentPlanner

logger = logging.getLogger(__name__)

class DecisionAgent:
    def __init__(self, gateway: AIGateway, settings: Any):
        self.gateway = gateway
        self.settings = settings

    async def run_optimization_loop(
        self, 
        dataset_path: str, 
        target_column: str, 
        n_hypotheses: int = 5, 
        max_workers: int = 3
    ) -> dict:
        logger.info(f"Starting DecisionAgent optimization loop on {dataset_path}")
        
        # 1. Profile the dataset
        loader = CsvLoader()
        df = loader.load(dataset_path)
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_column)

        planner = ExperimentPlanner(self.gateway)

        # Helper: create one experiment record in its own session
        async def create_experiment(create_data: ExperimentCreate) -> str:
            async with AsyncSessionLocal() as db:
                svc = ExperimentService(db)
                exp = await svc.create(create_data)
                await db.commit()
                return exp.id

        async def get_experiment_record(exp_id: str) -> tuple[dict, dict]:
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                from app.db.models.experiment import Experiment as ExpModel
                result = await db.execute(select(ExpModel).where(ExpModel.id == exp_id))
                exp = result.scalar_one_or_none()
                if not exp:
                    return {}, {}
                return exp.metrics or {}, exp.parameters or {}

        async def get_experiment_metrics(exp_id: str) -> dict:
            return (await get_experiment_record(exp_id))[0]

        async def save_decision(exp_id: str, decision: str, reason: str) -> None:
            from app.schemas.experiment import ExperimentUpdate
            async with AsyncSessionLocal() as db:
                await ExperimentService(db).update(
                    exp_id, ExperimentUpdate(decision=decision, decision_reason=reason)
                )
                await db.commit()

        # Baseline
        baseline_id = await create_experiment(ExperimentCreate(
            project_id="demo-project-id",
            dataset_version=dataset_path,
            hypothesis="Baseline without new features",
            change_description="Baseline run",
            model_name="XGBClassifier",
            feature_set=[],
            parameters={"target_column": target_column}
        ))
        # run_experiment_background opens its own session
        svc_for_run = ExperimentService(None)
        await svc_for_run.run_experiment_background(baseline_id)

        baseline_metrics = await get_experiment_metrics(baseline_id)
        # Decisions use validation F1; the test split is only for reporting.
        baseline_f1 = baseline_metrics.get("val_f1", 0.0)

        experiments_info = []
        winner_features = []
        # The champion is the best measured model so far: baseline, then each accepted
        # candidate. Every candidate = champion's features + one new feature.
        champion_id = baseline_id
        champion_features: list[dict] = []
        best_f1 = baseline_f1
        
        # History for the sequential loop
        history = [{"name": "baseline", "formula": "None", "f1": baseline_f1}]

        # 2 & 3. Sequential hypothesis generation and evaluation
        for i in range(n_hypotheses):
            try:
                hypothesis = await planner.generate_next_hypothesis(
                    profile=profile,
                    target_column=target_column,
                    objective="Maximize F1 score while preventing overfitting",
                    history=history
                )
            except Exception as e:
                logger.error(f"Failed to generate hypothesis: {e}")
                hypothesis = {
                    "name": f"fallback_feature_{i}", 
                    "formula": "feature * 1.5", 
                    "reason": "Fallback suggestion",
                    "non_redundant_reasoning": "Fallback"
                }
            
            # Create experiment
            eid = await create_experiment(ExperimentCreate(
                project_id="demo-project-id",
                parent_id=champion_id,
                dataset_version=dataset_path,
                hypothesis=hypothesis.get("reason", "Generated hypothesis"),
                change_description=f"Added feature: {hypothesis.get('name')} via formula {hypothesis.get('formula')}",
                model_name="XGBClassifier",
                feature_set=[hypothesis.get("name")] if hypothesis.get("name") else [],
                parameters={
                    "target_column": target_column,
                    "features": champion_features,
                    "feature_name": hypothesis.get("name"),
                    "formula": hypothesis.get("formula")
                }
            ))

            # Run experiment
            runner = ExperimentService(None)
            await runner.run_experiment_background(eid)
            result_metrics, result_params = await get_experiment_record(eid)
            result_f1 = result_metrics.get("val_f1", 0.0)

            feature_name = hypothesis.get("name", "unknown")
            formula = hypothesis.get("formula", "unknown")

            # The keep/reject decision is made by code (see ml/experiments/acceptance.py).
            # The LLM only explains it.
            acceptance = result_params.get("acceptance")
            invalid = result_params.get("invalid_formula")
            if invalid:
                decision = "reject"
                rule_summary = f"Rejected: the formula is invalid ({invalid['reason']}), so nothing was trained."
            elif acceptance is None:
                decision = "reject"
                rule_summary = "Rejected: the experiment did not complete, so there is no measured gain."
            else:
                decision = "keep" if acceptance["accepted"] else "reject"
                lo, hi = acceptance["ci95"]
                rule_summary = (
                    f"{'Accepted' if acceptance['accepted'] else 'Rejected'} by rule: mean "
                    f"{acceptance['metric']} gain {acceptance['mean_gain']:+.4f} "
                    f"(95% CI {lo:+.4f} to {hi:+.4f}) vs required margin {acceptance['margin']:.4f} "
                    f"over {len(acceptance['base_scores'])} paired folds."
                )

            explanation_mode = "llm"
            try:
                explanation = await self.gateway.complete(
                    task_type=TaskType.SUMMARIZE,
                    prompt=(
                        f"A candidate feature '{feature_name}' = {formula} was {decision}ed by a fixed "
                        f"statistical rule. Facts: {rule_summary} In two sentences, explain this "
                        "decision to a data analyst. Do not change or second-guess the decision."
                    ),
                    system="You explain ML experiment decisions plainly. Use only the facts given.",
                )
            except Exception as e:
                logger.error(f"Decision explanation failed: {e}")
                explanation, explanation_mode = "", "fallback"

            reason = rule_summary + (f" {explanation.strip()}" if explanation.strip() else "")
            await save_decision(eid, decision, reason)
            decision_data = {
                "decision": decision,
                "reason": reason,
                "decision_mode": "rule",
                "explanation_mode": explanation_mode,
                "acceptance": acceptance,
            }

            result_info = {
                "id": eid,
                "feature_name": feature_name,
                "formula": formula,
                "f1": result_f1,
                "decision": decision_data["decision"],
                "reason": decision_data["reason"],
                "decision_mode": decision_data["decision_mode"],
                "explanation_mode": decision_data["explanation_mode"],
                "acceptance": decision_data["acceptance"],
            }
            
            experiments_info.append(result_info)
            history.append(result_info)

            if result_info["decision"] == "keep":
                winner_features.append(result_info["feature_name"])
                champion_id = eid
                champion_features = champion_features + [{"name": feature_name, "formula": formula}]
                best_f1 = result_f1

        summary = (
            f"Tested {len(experiments_info)} features. Baseline validation F1: {baseline_f1:.4f}. "
            f"Champion validation F1: {best_f1:.4f}. Accepted: {', '.join(winner_features) if winner_features else 'None'}."
        )

        return {
            "winner_features": winner_features,
            "champion_id": champion_id,
            "champion_features": champion_features,
            "experiments": experiments_info,
            "best_f1": best_f1,
            "summary": summary
        }
