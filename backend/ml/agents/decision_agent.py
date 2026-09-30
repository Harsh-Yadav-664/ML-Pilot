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

import asyncio
import json
import logging
import re
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

        # 2. Generate hypotheses
        planner = ExperimentPlanner(self.gateway)
        try:
            hypotheses = await planner.generate_hypotheses(
                profile=profile,
                target_column=target_column,
                objective="Maximize F1 score while preventing overfitting",
                max_hypotheses=n_hypotheses
            )
        except Exception as e:
            logger.error(f"Failed to generate hypotheses: {e}")
            # Fallback to stub suggestions if AI Gateway fails
            hypotheses = [
                {
                    "name": f"fallback_feature_{i}", 
                    "formula": "feature * 1.5", 
                    "reason": "Fallback suggestion"
                } for i in range(n_hypotheses)
            ]

        async with AsyncSessionLocal() as db:
            svc = ExperimentService(db)

            # Baseline experiment
            baseline_create = ExperimentCreate(
                project_id="demo-project-id",
                dataset_version=dataset_path,
                hypothesis="Baseline without new features",
                change_description="Baseline run",
                model_name="XGBClassifier",
                feature_set=[],
                parameters={"target_column": target_column}
            )
            baseline_exp = await svc.create(baseline_create)
            await svc.run_experiment_background(baseline_exp.id)
            
            baseline_exp_db = await svc.get(baseline_exp.id)
            baseline_f1 = baseline_exp_db.metrics.get("f1", 0.0) if baseline_exp_db.metrics else 0.0

            experiments_info = []

            # 3. Create an experiment for each hypothesis
            exp_objects = []
            for hyp in hypotheses:
                exp_create = ExperimentCreate(
                    project_id="demo-project-id",
                    parent_id=baseline_exp.id,
                    dataset_version=dataset_path,
                    hypothesis=hyp.get("reason", "Generated hypothesis"),
                    change_description=f"Added feature: {hyp.get('name')} via formula {hyp.get('formula')}",
                    model_name="XGBClassifier",
                    feature_set=[hyp.get("name")] if hyp.get("name") else [],
                    parameters={
                        "target_column": target_column,
                        "feature_name": hyp.get("name"),
                        "formula": hyp.get("formula")
                    }
                )
                exp = await svc.create(exp_create)
                exp_objects.append((exp.id, hyp))

            # 3. Run all experiments in parallel
            sem = asyncio.Semaphore(max_workers)

            async def run_and_evaluate(exp_id: str, hypothesis: dict) -> dict:
                async with sem:
                    await svc.run_experiment_background(exp_id)
                    exp_db = await svc.get(exp_id)
                    result_f1 = exp_db.metrics.get("f1", 0.0) if exp_db.metrics else 0.0
                    
                    feature_name = hypothesis.get("name", "unknown")
                    formula = hypothesis.get("formula", "unknown")

                    prompt = (
                        f"Given baseline F1 of {baseline_f1} and this experiment got F1 of {result_f1}, "
                        f"feature: {feature_name}, formula: {formula}, should we keep this feature? "
                        "Reply with JSON: {\"decision\": \"keep\"|\"reject\", \"reason\": \"str\"}"
                    )
                    
                    try:
                        decision_str = await self.gateway.complete(
                            task_type=TaskType.DECIDE,
                            prompt=prompt,
                            system="You are an expert ML evaluator. Provide only raw valid JSON output."
                        )
                        # Remove markdown formatting if any
                        decision_str_clean = re.sub(r'```json|```', '', decision_str).strip()
                        decision_data = json.loads(decision_str_clean)
                    except Exception as e:
                        logger.error(f"Decision AI failed: {e}")
                        decision_data = {"decision": "keep" if result_f1 > baseline_f1 else "reject", "reason": f"Fallback decision (AI failed): {e}"}

                    return {
                        "id": exp_id,
                        "feature_name": feature_name,
                        "formula": formula,
                        "f1": result_f1,
                        "decision": decision_data.get("decision", "reject"),
                        "reason": decision_data.get("reason", "")
                    }

            results = await asyncio.gather(*(run_and_evaluate(e_id, h) for e_id, h in exp_objects))
            
            # 4 & 5. Evaluate results and summarize
            winner_features = []
            best_f1 = baseline_f1
            
            for res in results:
                experiments_info.append(res)
                if res["decision"] == "keep":
                    winner_features.append(res["feature_name"])
                if res["f1"] > best_f1:
                    best_f1 = res["f1"]

            summary = f"Tested {len(results)} features. Baseline F1: {baseline_f1:.4f}. Best F1: {best_f1:.4f}. Winners: {', '.join(winner_features) if winner_features else 'None'}."

            return {
                "winner_features": winner_features,
                "experiments": experiments_info,
                "best_f1": best_f1,
                "summary": summary
            }
