"""Experiment CRUD and orchestration service."""
from __future__ import annotations

import uuid
import logging
from typing import TYPE_CHECKING, Optional
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate
from ml.experiments.schema import ExperimentSpec, ExperimentStatus
from ml.experiments.executor import LocalExperimentExecutor
from ml.data.ingestion.csv_loader import CsvLoader

if TYPE_CHECKING:
    from ai.gateway import AIGateway

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
        """Run the experiment using the ML core and update the DB.
        
        IMPORTANT: Always opens its own AsyncSessionLocal so it never shares
        the request-scoped session (which is already closed by the time the
        background task runs), preventing SQLAlchemy 'session already closed' errors.
        """
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            exp = await session.get(Experiment, experiment_id)
            if not exp:
                logger.error(f"Experiment {experiment_id} not found for execution.")
                return

            exp.status = ExperimentStatus.RUNNING.value
            await session.commit()

            try:
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

                loader = CsvLoader()
                executor = LocalExperimentExecutor(data_loader_func=lambda p: loader.load(p))

                logger.info(f"Starting execution of experiment {experiment_id}...")
                result = await executor.run(spec)

                exp.status = result.status.value
                exp.metrics = result.metrics
                # New dict so SQLAlchemy persists the JSON change (best params, target encoding)
                exp.parameters = dict(result.parameters)
                exp.runtime_seconds = result.runtime_seconds
                exp.cost_usd = result.cost_usd
                exp.decision = result.decision.value
                logger.info(f"Experiment {experiment_id} completed. F1: {result.metrics.get('f1')}")

            except Exception as e:
                logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
                exp.status = ExperimentStatus.FAILED.value
                exp.decision_reason = str(e)

            finally:
                await session.commit()

    async def suggest_experiments(
        self,
        dataset_version: str,
        target_column: str,
        objective: str,
        max_hypotheses: int = 3,
        gateway: AIGateway | None = None,
    ) -> list[dict]:
        """Use the AI Gateway to generate feature hypotheses based on data.

        Raises ValueError if the dataset can't be loaded and RuntimeError if the
        planner fails.
        """
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
            raise ValueError(f"Could not load dataset {dataset_version}: {e}") from e

        # Profile the dataset deterministically
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_column)

        # The planner proposes one hypothesis at a time; feed earlier proposals
        # back as history so each new one is distinct.
        planner = ExperimentPlanner(gateway or AIGateway(settings))
        hypotheses: list[dict] = []
        try:
            for _ in range(max_hypotheses):
                history = [
                    {"name": h.get("name"), "formula": h.get("formula"), "status": "proposed"}
                    for h in hypotheses
                ]
                hypotheses.append(
                    await planner.generate_next_hypothesis(
                        profile=profile,
                        target_column=target_column,
                        objective=objective,
                        history=history,
                    )
                )
        except Exception as e:
            logger.error(f"AI Planner failed: {e}")
            raise RuntimeError(f"Failed to generate hypotheses: {e}") from e
        return hypotheses

