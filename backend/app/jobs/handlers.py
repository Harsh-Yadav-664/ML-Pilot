"""The kinds of work the job runner knows how to do.

Imported once at start-up (app.main) so the handlers are registered.
"""

from __future__ import annotations

from typing import Any

from ai.gateway import AIGateway
from app.core.config import settings
from app.db.models import Experiment
from app.jobs.runner import JobContext, handler, sessions
from app.services.experiment_service import ExperimentService
from ml.agents.decision_agent import DecisionAgent


def make_gateway() -> AIGateway:
    """The LLM gateway jobs use (tests replace it with the offline stub)."""
    return AIGateway(settings)


@handler("experiment")
async def run_experiment(ctx: JobContext, params: dict[str, Any]) -> dict[str, Any]:
    """Train one queued experiment (baseline, one feature, or an AI-cleaned baseline)."""
    exp_id = params["experiment_id"]
    await ctx.step("Training", 0.0, experiment_id=exp_id)
    await ExperimentService(None).run_experiment_background(exp_id)
    async with sessions()() as db:
        exp = await db.get(Experiment, exp_id)
    if exp is None:
        raise LookupError(f"Experiment {exp_id} disappeared while running")
    await ctx.emit("cv_result", experiment_id=exp_id, status=exp.status, metrics=exp.metrics or {})
    if exp.status == "failed":
        # The experiment row carries the reason too; the job must not read 'succeeded'.
        raise RuntimeError(exp.decision_reason or "experiment failed")
    return {"experiment_id": exp_id, "status": exp.status}


@handler("auto_optimize")
async def run_auto_optimize(ctx: JobContext, params: dict[str, Any]) -> dict[str, Any]:
    """The agent loop: baseline, then one hypothesis at a time, each decided by the rule."""
    agent = DecisionAgent(make_gateway(), settings)
    return await agent.run_optimization_loop(
        params["dataset_path"],
        params["target_column"],
        params["n_hypotheses"],
        project_id=params["project_id"],
        hooks=ctx,
    )
