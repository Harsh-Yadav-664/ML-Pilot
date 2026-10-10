"""A single-table run (CSV, Parquet): the agent job that runs on the one run loop (#151).

``execute`` is what the ``auto_optimize`` job does. It builds the baseline of the table
(``ml.features.table_baseline``: the existing preprocessing, a random split, LightGBM), then runs
``ml.agents.run_loop`` with a formula proposer. Acceptance is the loop's one paired rule and the
test rows are scored by the loop's one ``score_test``, once, at the end.

Everything is recorded as it happens, in short transactions of its own:

* a ``Run`` row without a task spec (a table has none), whose ``Feature`` rows are the proposals
  (kind ``formula``, with the stage that stopped one or its paired gain);
* the baseline is the first champion, an ``Experiment``; each accepted feature is a child
  experiment (the new champion, ``decision: keep``); a feature that did not help is a child with
  ``decision: reject``; a formula that failed a check is ``rejected_invalid`` (nothing was
  trained). Their metrics are validation metrics: the test rows are not scored before the end;
* at the end the last champion's row gets the test metrics, scored once. A cancelled run has none.

The job events ``proposal`` and ``decision`` are the ones the UI narrates.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from ai.gateway import AIGateway
from app.core import datasets
from app.core.config import settings
from app.db.models import Experiment, Run
from app.jobs import runner
from app.jobs.runner import JobCancelled, JobContext
from app.services.feature_proposals import feature_row
from app.services.privacy_service import builder_for
from ml.agents import run_loop as loop
from ml.agents.decision_explainer import explain_decision
from ml.agents.hooks import LoopHooks
from ml.data.profiling.profiler import DataProfiler
from ml.data.workspace import workspace_for
from ml.experiments.planner import ExperimentPlanner
from ml.features.baseline import LGBM_PARAMS, BaselineError, BaselineResult
from ml.features.formula_proposer import FormulaProposer
from ml.features.gain import FOLD_PARAMS
from ml.features.table_baseline import TableTask, build_table_baseline
from ml.metrics.calibration import choose_threshold
from ml.validation.splits import SplitError

ENGINE = "LGBMClassifier"
IMPORTANCE_METHOD = "LightGBM share of total gain (built-in importance), not SHAP"
SEED = 42

log = logging.getLogger(__name__)


def _importances(shares: dict[str, float], top_n: int = 10) -> dict[str, float]:
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: round(v, 6) for k, v in ranked}


def validation_metrics(champion: loop.Champion, y: np.ndarray, task: TableTask) -> dict[str, float]:
    """The champion's validation metrics: ranking metrics, and the class decisions at the
    threshold that is best on the same validation rows (the test rows play no part)."""
    val = task.split.val
    prob = np.asarray(champion.model.predict_proba(champion.frame.iloc[val]))[:, 1]
    threshold = float(choose_threshold(y[val], prob)["value"])
    metrics: dict[str, float | None] = {
        **champion.metrics,
        **loop.classification_at(y[val], prob, threshold),
    }
    return {k: float(v) for k, v in metrics.items() if v is not None}


class _Sink:
    """Writes each round as it ends: the proposal, and the experiment that records its outcome."""

    def __init__(
        self,
        ctx: LoopHooks,
        run_id: str,
        parent: str,
        *,
        project_id: str,
        dataset_path: str,
        data_version_id: str | None,
        task: TableTask,
        baseline: BaselineResult,
        gateway: AIGateway,
        builder: Any,
        seed: int,
    ) -> None:
        self.ctx = ctx
        self.run_id = run_id
        self.parent = parent  # the current champion's experiment
        self.project_id = project_id
        self.dataset_path = dataset_path
        self.data_version_id = data_version_id
        self.task = task
        self.baseline = baseline
        self.gateway = gateway
        self.builder = builder
        self.seed = seed
        self.position = 0
        self.champion_features: list[dict[str, str]] = []  # name and formula, in order added
        self.infos: list[dict[str, Any]] = []  # one per proposal, for the job result
        self.y = np.asarray(task.y)

    # -- what every experiment of the run records ----------------------------------------------
    def parameters(self, champion: loop.Champion, **more: Any) -> dict[str, Any]:
        t = self.task
        return {
            "target_column": t.target_column,
            "positive_class": t.encoder.positive_class,
            "target_encoding": t.encoder.to_dict(),
            "excluded_features": t.report["excluded_features"],
            "numeric_coercion": t.report["numeric_coercion"],
            "split_plan": t.split.plan.model_dump(),
            "split": t.split.summary(),
            "feature_columns": champion.names,
            "best_params": {**LGBM_PARAMS, "n_estimators": max(10, champion.best_iteration)},
            "engine": {"name": ENGINE, "version": lgb.__version__},
            "feature_importances": _importances(champion.shares),
            "importance_method": IMPORTANCE_METHOD,
            "tuning": "none: fixed LightGBM parameters, stopped early on the validation rows",
            "run_loop": {"run_id": self.run_id, "fold_kind": self.baseline.fold_kind},
            **more,
        }

    def manifest(self, kind: str, **more: Any) -> dict[str, Any]:
        """What is needed to reproduce the row; named fields only, no prompt text, no secrets."""
        return {
            "kind": kind,
            "data_version_id": self.data_version_id,
            "target_column": self.task.target_column,
            "split": self.task.split.summary(),
            "seed": self.seed,
            "engine": {"name": ENGINE, "version": lgb.__version__, "params": LGBM_PARAMS},
            "features": list(self.champion_features),
            "decision_mode": "rule",
            "mlpilot_version": settings.APP_VERSION,
            **more,
        }

    async def round_done(self, rnd: loop.Round, champion: loop.Champion) -> None:
        self.position += 1
        record = rnd.record
        p = record.proposal
        meta = record.llm[0] if record.llm else {}
        await self.ctx.emit(
            "proposal",
            index=rnd.number,
            name=p.name if p else None,
            formula=p.formula if p else None,
            decision_mode=meta.get("decision_mode"),
        )
        exp_id: str | None = None
        decision = "reject"
        info: dict[str, Any]
        if p is None:
            # nothing usable came back: only the Feature row is kept, there is no experiment
            reason = f"Rejected: the proposal could not be read ({'; '.join(record.reasons)})."
            async with runner.sessions()() as db:
                db.add(feature_row(self.run_id, record, self.position, None))
                await db.commit()
            info = self._info(None, rnd, "reject", reason, None, meta)
            self.infos.append(info)
            await self._decision_event(rnd, None, "reject", reason)
            return
        if rnd.gain is None:
            # stopped by a check: the formula is invalid or a duplicate, nothing was trained
            why = "; ".join(record.reasons)
            rule_summary = (
                f"Rejected: the formula is invalid ({why}), so nothing was trained."
                if record.stage in ("guard", "execution", "schema")
                else f"Rejected at the {record.stage} check: {why}"
            )
            explanation, mode = await explain_decision(
                self.gateway,
                self.builder,
                name=p.name,
                formula=p.formula,
                decision="reject",
                facts=rule_summary,
            )
            reason = rule_summary + (f" {explanation.strip()}" if explanation.strip() else "")
            exp_id = await self._write(
                rnd,
                champion,
                status="rejected_invalid",
                decision="reject",
                reason=reason,
                gain=None,
                metrics=None,
                invalid={"name": p.name, "formula": p.formula, "reason": why},
                meta=meta,
            )
        else:
            gain = rnd.gain
            lo, hi = gain.ci95
            decision = "keep" if gain.accepted else "reject"
            rule_summary = (
                f"{'Accepted' if gain.accepted else 'Rejected'} by rule: mean {gain.metric} gain "
                f"{gain.mean_gain:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}) vs required margin "
                f"{gain.margin:.4f} over {len(gain.base_scores)} paired folds."
            )
            explanation, mode = await explain_decision(
                self.gateway,
                self.builder,
                name=p.name,
                formula=p.formula,
                decision=decision,
                facts=rule_summary,
            )
            reason = rule_summary + (f" {explanation.strip()}" if explanation.strip() else "")
            metrics = (
                await asyncio.to_thread(validation_metrics, champion, self.y, self.task)
                if rnd.accepted
                else None
            )
            exp_id = await self._write(
                rnd,
                champion,
                status="completed",
                decision=decision,
                reason=reason,
                gain=gain.to_dict(),
                metrics=metrics,
                invalid=None,
                meta=meta,
            )
        info = self._info(exp_id, rnd, decision, reason, mode, meta)
        self.infos.append(info)
        if rnd.accepted and p.formula is not None:
            self.champion_features.append({"name": p.name, "formula": p.formula})
        await self._decision_event(rnd, exp_id, decision, rule_summary)

    def _info(
        self,
        exp_id: str | None,
        rnd: loop.Round,
        decision: str,
        reason: str,
        mode: str | None,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        p = rnd.record.proposal
        return {
            "id": exp_id,
            "feature_name": p.name if p else None,
            "formula": p.formula if p else None,
            "decision": decision,
            "reason": reason,
            "decision_mode": "rule",
            "explanation_mode": mode,
            "acceptance": rnd.gain.to_dict() if rnd.gain else None,
            "hypothesis_llm": meta,
        }

    async def _decision_event(
        self, rnd: loop.Round, exp_id: str | None, decision: str, summary: str
    ) -> None:
        p = rnd.record.proposal
        await self.ctx.emit(
            "decision",
            experiment_id=exp_id,
            name=p.name if p else None,
            decision=decision,
            decision_mode="rule",
            reason=summary,
        )

    async def _write(
        self,
        rnd: loop.Round,
        champion: loop.Champion,
        *,
        status: str,
        decision: str,
        reason: str,
        gain: dict[str, Any] | None,
        metrics: dict[str, float] | None,
        invalid: dict[str, Any] | None,
        meta: dict[str, Any],
    ) -> str:
        p = rnd.record.proposal
        assert p is not None
        # a rejected candidate is described as the champion that stood when it was tried plus
        # itself; an accepted one is the new champion, whose parameters list its features
        before = list(self.champion_features)
        extra: dict[str, Any] = {
            "features": before,
            "feature_name": p.name,
            "formula": p.formula,
            "hypothesis_llm": meta,
        }
        if gain is not None:
            extra["acceptance"] = {**gain, "model": {"model": ENGINE, **FOLD_PARAMS}}
        if invalid is not None:
            extra["invalid_formula"] = invalid
        params = self.parameters(champion, **extra)
        if not rnd.accepted:
            # the candidate was not adopted: its champion is the previous one
            params["feature_columns"] = [*champion.names, p.name] if gain else champion.names
        exp = Experiment(
            id=str(uuid.uuid4()),
            parent_id=self.parent,
            project_id=self.project_id,
            dataset_version=self.dataset_path,
            hypothesis=p.rationale or "Generated hypothesis",
            change_description=f"Added feature: {p.name} via formula {p.formula}",
            model_name=ENGINE,
            feature_set=[p.name],
            parameters=params,
            validation_config=self.task.split.plan.model_dump(),
            metrics=metrics if metrics is not None else {},
            val_metrics=metrics,
            status=status,
            decision=decision,
            decision_reason=reason,
            decision_mode="rule",
            decision_detail=gain,
            runtime_seconds=rnd.seconds,
            run_id=self.run_id,
            data_version_id=self.data_version_id,
            manifest=self.manifest(
                "loop_champion" if rnd.accepted else "loop_candidate",
                round=rnd.number,
                added={"name": p.name, "formula": p.formula},
                llm=[_llm_entry(m) for m in rnd.record.llm],
            ),
        )
        async with runner.sessions()() as db:
            db.add(feature_row(self.run_id, rnd.record, self.position, rnd.gain, rnd.vetoed))
            db.add(exp)
            await db.flush()
            if rnd.accepted:
                self.parent = exp.id
            await db.commit()
        return exp.id


def _llm_entry(meta: dict[str, Any]) -> dict[str, Any]:
    """The provider, model, tokens, cost and mode of one call; never the text."""
    keys = ("provider", "model", "tokens_in", "tokens_out", "cost_usd", "decision_mode")
    return {k: meta.get(k) for k in keys} | {"providers_failed": len(meta.get("errors") or [])}


async def _mark(run_id: str, status: str, error: str | None = None) -> None:
    async with runner.sessions()() as db:
        run = await db.get(Run, run_id)
        if run is not None:
            run.status, run.finished_at = status, datetime.now(UTC)
            if error:
                run.error = error[:2000]
            await db.commit()


async def execute(
    ctx: JobContext, params: dict[str, Any], make_gateway: Any = None
) -> dict[str, Any]:
    """The job: baseline, the run loop, the champion's test metrics. Fails loudly on an error."""
    gateway: AIGateway = (make_gateway or _default_gateway)()
    run_id = str(uuid.uuid4())
    try:
        return await _execute(ctx, run_id, params, gateway)
    except JobCancelled:
        await _mark(
            run_id, "cancelled", "cancelled by request; the champion recorded so far is kept"
        )
        raise
    except (BaselineError, SplitError) as e:
        await _mark(run_id, "failed", str(e))
        raise RuntimeError(str(e)) from e
    except Exception as e:
        await _mark(run_id, "failed", f"{type(e).__name__}: {e}")
        raise


def _default_gateway() -> AIGateway:
    from app.jobs.handlers import make_gateway

    return make_gateway()


async def _execute(
    ctx: JobContext, run_id: str, params: dict[str, Any], gateway: AIGateway
) -> dict[str, Any]:
    project_id = params["project_id"]
    dataset_path = params["dataset_path"]
    target = params["target_column"]
    rounds = int(params["n_hypotheses"])
    data_version_id = params.get("data_version_id")
    seed = SEED

    await ctx.step("Profiling the data and training the baseline", 0.0)
    workspace = workspace_for(project_id, datasets.PROJECTS_DIR)
    df = await asyncio.to_thread(workspace.load, Path(dataset_path))
    profile = DataProfiler().profile(df, target_column=target)
    builder = await builder_for(project_id)
    baseline, task = await asyncio.to_thread(build_table_baseline, df, target, seed=seed)
    y = np.asarray(task.y)

    config = loop.RunConfig(max_rounds=rounds, patience=rounds, seed=seed)
    state, scorer = await asyncio.to_thread(loop.start_state, baseline, config)
    base_metrics = await asyncio.to_thread(validation_metrics, state.champion, y, task)

    run = Run(
        id=run_id,
        project_id=project_id,
        task_spec_id=None,  # a table has no task spec
        data_version_id=data_version_id,
        split_plan=task.split.plan.model_dump(),
        engine=ENGINE,
        seed=seed,
        status="running",
        started_at=datetime.now(UTC),
        manifest={"kind": "table_run", "target_column": target, "dataset": Path(dataset_path).name},
        budget={"max_rounds": rounds, "patience": rounds},
    )
    sink = _Sink(
        ctx,
        run_id,
        "",
        project_id=project_id,
        dataset_path=dataset_path,
        data_version_id=data_version_id,
        task=task,
        baseline=baseline,
        gateway=gateway,
        builder=builder,
        seed=seed,
    )
    baseline_exp = Experiment(
        id=str(uuid.uuid4()),
        parent_id=None,
        project_id=project_id,
        dataset_version=dataset_path,
        hypothesis="Baseline without new features",
        change_description="Baseline run",
        model_name=ENGINE,
        feature_set=[],
        parameters=sink.parameters(state.champion, features=[]),
        validation_config=task.split.plan.model_dump(),
        metrics=base_metrics,
        val_metrics=base_metrics,
        status="completed",
        decision="pending",
        decision_mode="rule",
        runtime_seconds=baseline.seconds,
        run_id=run_id,
        data_version_id=data_version_id,
        manifest=sink.manifest("loop_baseline"),
    )
    async with runner.sessions()() as db:
        db.add(run)
        await db.flush()
        db.add(baseline_exp)
        await db.flush()
        run.champion_experiment_id = baseline_exp.id
        await db.commit()
    sink.parent = baseline_exp.id
    await ctx.emit(
        "cv_result",
        experiment_id=baseline_exp.id,
        role="baseline",
        val_pr_auc=base_metrics.get("pr_auc"),
        val_f1=base_metrics.get("f1"),
    )

    proposer = FormulaProposer(
        planner=ExperimentPlanner(gateway, builder),
        profile=profile,
        task=task,
        baseline=baseline,
    )
    outcome = await loop.run_loop(
        baseline,
        proposer,
        config,
        hooks=ctx,
        sink=sink,
        state=state,
        scorer=scorer,
    )
    return await _finish(run_id, outcome, sink, baseline_exp.id, base_metrics, y, task)


async def _finish(
    run_id: str,
    outcome: loop.RunOutcome,
    sink: _Sink,
    baseline_id: str,
    base_metrics: dict[str, float],
    y: np.ndarray,
    task: TableTask,
) -> dict[str, Any]:
    """The champion's row gets the test metrics (scored once, by the loop); the run is closed."""
    champion_metrics = await asyncio.to_thread(validation_metrics, outcome.champion, y, task)
    now = datetime.now(UTC)
    async with runner.sessions()() as db:
        exp = await db.get(Experiment, sink.parent)
        assert exp is not None
        exp.val_metrics = champion_metrics
        if outcome.test is not None:
            # as for every experiment before: unprefixed keys are the test metrics, with the
            # test_ and val_ copies beside them
            exp.test_metrics = outcome.test
            exp.metrics = {
                **outcome.test,
                **{f"test_{k}": v for k, v in outcome.test.items()},
                **{f"val_{k}": v for k, v in champion_metrics.items()},
            }
        exp.manifest = {
            **(exp.manifest or {}),
            "final_model": "the champion's features refit on train and validation; test scored once",
            "test_error": outcome.test_error,
            "stop_reason": outcome.stop_reason,
        }
        run = await db.get(Run, run_id)
        assert run is not None
        run.status = outcome.status
        run.finished_at = now
        run.champion_experiment_id = exp.id
        run.budget_used = {**outcome.used, "stopped": outcome.stop_reason}
        run.manifest = {
            **run.manifest,
            "run_loop": {
                "stop_reason": outcome.stop_reason,
                "rounds": len(outcome.rounds),
                "accepted": [r.record.proposal.name for r in outcome.accepted if r.record.proposal],
                "test_error": outcome.test_error,
                "at": now.isoformat(),
            },
        }
        await db.commit()
    kept = [i["feature_name"] for i in sink.infos if i["decision"] == "keep"]
    base_pr = base_metrics.get("pr_auc")
    best_pr = champion_metrics.get("pr_auc")
    summary = (
        f"Tested {len(sink.infos)} features. Baseline validation PR-AUC: "
        f"{base_pr:.4f}. Champion validation PR-AUC: {best_pr:.4f}. "
        f"Accepted: {', '.join(kept) if kept else 'None'}."
        if base_pr is not None and best_pr is not None
        else f"Tested {len(sink.infos)} features. Accepted: {', '.join(kept) if kept else 'None'}."
    )
    return {
        "run_id": run_id,
        "status": outcome.status,
        "stop_reason": outcome.stop_reason,
        "winner_features": kept,
        "champion_id": sink.parent,
        "champion_features": list(sink.champion_features),
        "experiments": sink.infos,
        "best_f1": champion_metrics.get("f1"),
        "best_pr_auc": best_pr,
        "test": outcome.test,
        "test_error": outcome.test_error,
        "summary": summary,
    }
