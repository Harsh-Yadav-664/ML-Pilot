"""Experiment CRUD and orchestration service."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.core.config import settings
from app.db.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate
from ml.data.versions import content_hash, version_id_of
from ml.data.workspace import workspace_for
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.manifest import RunManifest, build_manifest, replay
from ml.experiments.schema import ExperimentResult, ExperimentSpec, ExperimentStatus

if TYPE_CHECKING:
    from ai.gateway import AIGateway

logger = logging.getLogger(__name__)


class ExperimentService:
    def __init__(self, db: AsyncSession | None) -> None:
        # None for callers that only use the methods which open their own session.
        self._db = db

    @property
    def db(self) -> AsyncSession:
        if self._db is None:
            raise RuntimeError("ExperimentService was created without a database session")
        return self._db

    async def create(self, data: ExperimentCreate) -> Experiment:
        fields = data.model_dump()
        if fields.get("data_version_id") is None:
            fields["data_version_id"] = version_id_of(data.dataset_version, datasets.VERSIONS_DIR)
        experiment = Experiment(id=str(uuid.uuid4()), **fields)
        self.db.add(experiment)
        await self.db.flush()
        return experiment

    async def get(self, experiment_id: str) -> Experiment | None:
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

    async def update(self, experiment_id: str, data: ExperimentUpdate) -> Experiment | None:
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
                # Nothing to mark as failed; raise so the caller logs it loudly.
                raise LookupError(f"Experiment {experiment_id} not found for execution")

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

                workspace = workspace_for(exp.project_id, datasets.PROJECTS_DIR)
                executor = LocalExperimentExecutor(data_loader_func=workspace.load)

                logger.info(f"Starting execution of experiment {experiment_id}...")
                result = await executor.run(spec)

                exp.status = result.status.value
                exp.metrics = result.metrics
                # New dict so SQLAlchemy persists the JSON change (best params, target encoding)
                exp.parameters = dict(result.parameters)
                exp.runtime_seconds = result.runtime_seconds
                exp.cost_usd = result.cost_usd
                exp.decision = result.decision.value
                if result.decision_reason:
                    exp.decision_reason = result.decision_reason
                if result.status == ExperimentStatus.COMPLETED:
                    exp.manifest = build_manifest(
                        result,
                        data_version_id=exp.data_version_id,
                        preprocessing_config=exp.preprocessing_config,
                        mlpilot_version=settings.APP_VERSION,
                    ).model_dump(mode="json")
                logger.info(f"Experiment {experiment_id} completed. F1: {result.metrics.get('f1')}")

            except Exception as e:
                logger.exception(f"Experiment {experiment_id} failed")
                exp.status = ExperimentStatus.FAILED.value
                exp.decision_reason = str(e)

            finally:
                await session.commit()

    async def replay(self, experiment_id: str) -> ExperimentResult:
        """Retrain and re-score a completed experiment from its manifest alone: same data
        version (checked by content hash), features, parameters, split and seed; no LLM."""
        exp = await self.get(experiment_id)
        if exp is None or exp.manifest is None:
            raise LookupError(f"Experiment {experiment_id} has no run manifest to replay")
        manifest = RunManifest.model_validate(exp.manifest)
        if manifest.data_version_id is None:
            raise LookupError(f"Experiment {experiment_id} did not train on a stored data version")
        path = datasets.VERSIONS_DIR / f"{manifest.data_version_id}.csv"
        if not path.exists() or content_hash(path) != manifest.data_version_id:
            raise LookupError(f"Data version {manifest.data_version_id} is missing or changed")
        workspace = workspace_for(exp.project_id, datasets.PROJECTS_DIR)
        return await replay(manifest, str(path), workspace.load)

    async def suggest_experiments(
        self,
        dataset_version: str,
        project_id: str,
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
        from ml.data.profiling.profiler import DataProfiler
        from ml.experiments.planner import ExperimentPlanner

        # In MVP, assume dataset_version is a valid local file path
        try:
            df = workspace_for(project_id, datasets.PROJECTS_DIR).load(Path(dataset_version))
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


def champion_path(experiments: list) -> list:
    """Return the champion lineage: the latest completed baseline, then each kept child.

    The last element is the champion, i.e. the best model that was actually measured
    with all accepted features. Empty when there is no completed baseline.
    """
    roots = [e for e in experiments if e.parent_id is None and e.status == "completed"]
    if not roots:
        return []
    path = [max(roots, key=lambda e: e.created_at)]
    while True:
        kept = [
            e
            for e in experiments
            if e.parent_id == path[-1].id and e.decision == "keep" and e.status == "completed"
        ]
        if not kept:
            return path
        path.append(max(kept, key=lambda e: e.created_at))
