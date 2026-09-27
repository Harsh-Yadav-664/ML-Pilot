"""Experiment CRUD and orchestration service."""
from __future__ import annotations

import uuid
import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate
from ml.experiments.schema import ExperimentSpec, ExperimentStatus
from ml.experiments.executor import LocalExperimentExecutor
from ml.data.ingestion.csv_loader import CsvLoader

logger = logging.getLogger(__name__)

class ExperimentService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: ExperimentCreate) -> Experiment:
        experiment = Experiment(id=str(uuid.uuid4()), **data.model_dump())
        self.db.add(experiment)
        await self.db.flush()
        return experiment

    async def get(self, experiment_id: str) -> Optional[Experiment]:
        result = await self.db.execute(select(Experiment).where(Experiment.id == experiment_id))
        return result.scalar_one_or_none()

    async def list_by_project(
        self, project_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Experiment], int]:
        offset = (page - 1) * page_size
        count_q = await self.db.execute(
            select(func.count()).select_from(Experiment).where(Experiment.project_id == project_id)
        )
        total = count_q.scalar_one()
        result = await self.db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id)
            .offset(offset)
            .limit(page_size)
            .order_by(Experiment.created_at.desc())
        )
        return list(result.scalars().all()), total

    async def update(self, experiment_id: str, data: ExperimentUpdate) -> Optional[Experiment]:
        exp = await self.get(experiment_id)
        if not exp:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(exp, field, value)
        await self.db.flush()
        return exp

    async def run_experiment_background(self, experiment_id: str) -> None:
        """Run the experiment using the ML core and update the DB."""
        # Refresh the session for the background task
        exp = await self.get(experiment_id)
        if not exp:
            logger.error(f"Experiment {experiment_id} not found for execution.")
            return

        exp.status = ExperimentStatus.RUNNING.value
        await self.db.commit()

        try:
            # 1. Convert ORM to ML Spec
            spec = ExperimentSpec(
                id=exp.id,
                parent_id=exp.parent_id,
                project_id=exp.project_id,
                dataset_version=exp.dataset_version,
                hypothesis=exp.hypothesis,
                change_description=exp.change_description,
                model_name=exp.model_name,
                parameters=exp.parameters,
                validation_config=exp.validation_config,
                feature_set=exp.feature_set,
                preprocessing_config=exp.preprocessing_config,
                budget=exp.budget,
            )

            # 2. Configure the Executor
            def load_dataset(version_path: str):
                # For MVP, assume dataset_version is a valid local file path (e.g. data.csv)
                # In prod, this would download from S3 based on ID
                loader = CsvLoader()
                return loader.load(version_path)

            executor = LocalExperimentExecutor(data_loader_func=load_dataset)

            # 3. Execute
            logger.info(f"Starting execution of experiment {experiment_id}...")
            result = await executor.run(spec)

            # 4. Update DB Record
            exp.status = result.status.value
            exp.metrics = result.metrics
            exp.runtime_seconds = result.runtime_seconds
            exp.cost_usd = result.cost_usd
            exp.decision = result.decision.value
            
            logger.info(f"Experiment {experiment_id} completed successfully. F1: {result.metrics.get('f1')}")

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
            exp.status = ExperimentStatus.FAILED.value
            exp.decision_reason = str(e)
        
        finally:
            await self.db.commit()

    async def suggest_experiments(self, dataset_version: str, target_column: str, objective: str, max_hypotheses: int = 3) -> list[dict]:
        """Use the AI Gateway to generate feature hypotheses based on data."""
        from ai.gateway import AIGateway
        from app.core.config import settings
        from ml.experiments.planner import ExperimentPlanner
        from ml.data.profiling.profiler import DataProfiler

        # In MVP, assume dataset_version is a valid local file path
        loader = CsvLoader()
        try:
            df = loader.load(dataset_version)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_version} for suggestion: {e}")
            raise ValueError(f"Could not load dataset {dataset_version}")

        # Profile the dataset deterministically
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_column)

        # Generate hypotheses via AI
        gateway = AIGateway(settings)
        planner = ExperimentPlanner(gateway)
        
        try:
            hypotheses = await planner.generate_hypotheses(
                profile=profile,
                target_column=target_column,
                objective=objective,
                max_hypotheses=max_hypotheses
            )
            return hypotheses
        except Exception as e:
            logger.error(f"AI Planner failed: {e}")
            raise RuntimeError(f"Failed to generate hypotheses: {e}")

