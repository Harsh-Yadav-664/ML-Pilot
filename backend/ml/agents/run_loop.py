"""The one loop of a run (#58, #151): baseline, propose, check, execute, accept by gain, repeat.

It runs relational tasks (temporal folds, SQL features) and single-table tasks (random folds,
formula features, ``ml/features/table_baseline.py``) alike; only the baseline, the proposer and
the kind of folds differ. The text below describes the relational case.

    champion = the baseline's features and model                       (#55)
    each round:
        stop if a budget is reached, the rounds are used up or ``patience`` rounds passed
        without an accepted feature
        ask the proposer for one feature                                (#56)
        it is guard-checked, executed and de-duplicated by the proposer (#57)
        score champion and champion + feature on the same temporal folds of the training rows
        accept in code if the paired gain beats the margin              (ml/features/gain.py)
    at the end: refit the champion on train + validation and score the test rows once.

The language model only proposes. It never sees a score decision, and a feature that does not
help is kept out by the rule above whatever the model says about it. The test rows are scored by
``score_test`` exactly once, after the last round; a cancelled run is not scored at all.

The loop persists nothing itself. After each round it calls ``sink.round_done`` (the caller
writes the feature row and, for an accepted feature, a new champion), so an interrupted run
always has its last champion on record.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.agents.hooks import LoopHooks, NoHooks
from ml.experiments.acceptance import GainResult
from ml.features.baseline import (
    LGBM_PARAMS,
    BaselineError,
    BaselineResult,
    fit_validated,
)
from ml.features.engine import Budget, BudgetTracker
from ml.features.gain import FoldScorer, compare
from ml.features.ir import describe
from ml.features.llm_sql import ProposalRecord
from ml.metrics.calibration import choose_threshold
from ml.metrics.classification import compute_classification_metrics
from ml.metrics.ranking import ranking_metrics
from ml.validation.splits import RowSplit, SplitError

RunStatus = Literal["completed", "stopped", "cancelled"]


ApprovalMode = Literal["auto", "confirm_task", "approve_each_feature"]


@dataclass(frozen=True)
class RunConfig:
    max_rounds: int = 20  # proposals asked for
    patience: int = 5  # stop after this many rounds in a row without an accepted feature
    budget: Budget = field(default_factory=Budget)
    rule: dict[str, Any] | None = None  # overrides of ml.experiments.acceptance.DEFAULT_RULE
    seed: int = 42
    # auto and confirm_task never pause the loop (the task was confirmed before the run);
    # approve_each_feature asks the person before every proposal that passed the checks
    approval_mode: ApprovalMode = "confirm_task"


@dataclass
class Champion:
    """The feature set the run currently stands on."""

    names: list[str]
    frame: pd.DataFrame = field(repr=False)
    metrics: dict[str, float]  # validation metrics of its model
    shares: dict[str, float]  # share of gain per feature
    best_iteration: int
    fold_scores: list[float]  # PR-AUC per temporal fold, the bar for the next candidate
    model: Any = field(repr=False, default=None)


@dataclass
class Round:
    number: int
    record: ProposalRecord
    gain: GainResult | None = None  # None if the proposal never reached the gain test
    accepted: bool = False
    champion_metrics: dict[str, float] | None = None  # validation metrics, set when accepted
    seconds: float = 0.0
    vetoed: bool = False  # the person said no at a checkpoint: it never reached the gain test
    checkpoint: dict[str, Any] | None = None  # how the question was answered


@dataclass
class LoopState:
    """Updated as the loop goes, so a caller that is interrupted can see where it stood."""

    champion: Champion
    rounds: list[Round] = field(default_factory=list)
    tracker: BudgetTracker | None = None


@dataclass
class RunOutcome:
    status: RunStatus
    stop_reason: str
    rounds: list[Round]
    champion: Champion
    test: dict[str, float] | None
    test_error: str | None
    used: dict[str, Any]

    @property
    def accepted(self) -> list[Round]:
        return [r for r in self.rounds if r.accepted]


@dataclass(frozen=True)
class Answer:
    decision: Literal["approve", "veto"]
    how: Literal["reply", "timeout"]  # a timeout continues with the recommended action


class Proposer(Protocol):
    """What the loop needs of whatever proposes features: ``FeatureProposer`` (SQL features of a
    relational task) or ``FormulaProposer`` (formulas over the columns of one table)."""

    async def propose(self, remaining: int = 1, hints: Sequence[str] = ()) -> ProposalRecord: ...

    def adopt(self, record: ProposalRecord, importance: float = 0.0) -> None: ...

    def reject_for_gain(self, record: ProposalRecord, reason: str, stage: str = "gain") -> None: ...


class Control(Protocol):
    """What a person can do to a running loop: change its settings, suggest, approve or veto.

    The loop only asks; the answers come from the database (``app/services/run_service.py``).
    """

    async def settings(self) -> RunConfig | None:
        """The settings as they are now, read before every round (None: unchanged)."""

    async def hints(self, number: int) -> list[str]:
        """Suggestions not yet used; they are marked as used in round ``number``."""

    async def approve(self, number: int, summary: dict[str, Any]) -> Answer:
        """Wait for the person to approve or veto the proposal (or for the timeout)."""


class Sink(Protocol):
    async def round_done(self, rnd: Round, champion: Champion) -> None: ...


class NoSink:
    async def round_done(self, rnd: Round, champion: Champion) -> None:
        return None


def score_test(
    model: Any, features: pd.DataFrame, y: np.ndarray, *, threshold: float | None = None
) -> dict[str, float]:
    """Score the test rows. The one place the test rows are scored; called once per run.

    With a ``threshold`` (chosen on the validation rows, never here) the same predictions are
    also reported as class decisions: accuracy, precision, recall, F1 and ROC-AUC.
    """
    prob = np.asarray(model.predict_proba(features))[:, 1]
    metrics: dict[str, float | None] = dict(ranking_metrics(y, prob, absolute_k=(100,)))
    metrics["n_test"] = float(len(y))
    if threshold is not None:
        metrics.update(classification_at(y, prob, threshold))
    return {k: float(v) for k, v in metrics.items() if v is not None}


def classification_at(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | None]:
    """Class decisions at a threshold (accuracy, precision, recall, F1) and ROC-AUC."""
    both = np.column_stack([1.0 - prob, prob])
    out = compute_classification_metrics(y, (prob >= threshold).astype(int), both)
    out["threshold"] = float(threshold)
    return out


def start_state(baseline: BaselineResult, config: RunConfig) -> tuple[LoopState, FoldScorer]:
    """The champion at the start of the loop: the baseline, scored on the temporal folds."""
    if (
        baseline.frame is None
        or baseline.labels is None
        or baseline.temporal is None
        or baseline.model is None
    ):
        raise BaselineError("the baseline result must carry its frame, labels, model and split")
    y = baseline.labels["label"].astype(int).to_numpy()
    scorer = FoldScorer(
        y,
        baseline.temporal.folds,
        train_rows=baseline.temporal.fold_rows,
        seed=config.seed,
        kind=baseline.fold_kind,
    )
    champion = Champion(
        names=[str(c) for c in baseline.frame.columns],
        frame=baseline.frame,
        metrics=baseline.metrics,
        shares={f.candidate.name: f.importance for f in baseline.features},
        best_iteration=int(baseline.metrics["best_iteration"]),
        fold_scores=scorer.score(baseline.frame),
        model=baseline.model,
    )
    return LoopState(champion), scorer


async def run_loop(
    baseline: BaselineResult,
    proposer: Proposer,
    config: RunConfig,
    *,
    hooks: LoopHooks | None = None,
    sink: Sink | None = None,
    state: LoopState | None = None,
    scorer: FoldScorer | None = None,
    control: Control | None = None,
    score_test: bool = True,
) -> RunOutcome:
    """Run the rounds, then score the test rows once. See the module docstring."""
    hooks = hooks or NoHooks()
    sink = sink or NoSink()
    if state is None or scorer is None:
        started_state, started_scorer = await asyncio.to_thread(start_state, baseline, config)
        state, scorer = state or started_state, scorer or started_scorer
    assert state is not None and scorer is not None
    assert baseline.labels is not None and baseline.temporal is not None
    y = baseline.labels["label"].astype(int).to_numpy()
    split = baseline.temporal
    tracker = state.tracker if state.tracker is not None else BudgetTracker(config.budget)
    state.tracker = tracker
    stop_reason = "max_rounds"
    since_accept = 0
    await hooks.step("Starting from the baseline", 0.0, champion=state.champion.metrics)
    number = 0
    while True:
        number += 1
        await hooks.check_cancelled()
        if control is not None:  # settings may have changed since the last round
            config = await control.settings() or config
            tracker.budget = config.budget
        if number > config.max_rounds:
            break
        why = tracker.exceeded()
        if why:
            stop_reason = f"budget:{why}"
            break
        if since_accept >= config.patience:
            stop_reason = "patience"
            break
        started = time.monotonic()
        await hooks.step(f"Round {number}: asking for a feature", (number - 1) / config.max_rounds)
        hints = await control.hints(number) if control is not None else []
        record = await proposer.propose(config.max_rounds - number + 1, hints=hints)
        tracker.proposals += 1
        cost = sum(float(c.get("cost_usd") or 0.0) for c in record.llm)
        tracker.charge(cost)
        if record.status == "no_llm":
            stop_reason = "no_llm"
            rnd = Round(number, record, seconds=time.monotonic() - started)
            await hooks.emit("feature_decision", **_event(rnd, state.champion, tracker, cost))
            break
        rnd = Round(number, record)
        if record.status == "proposed" and record.proposal is not None:
            if control is not None and config.approval_mode == "approve_each_feature":
                answer = await control.approve(number, _summary(record))
                rnd.checkpoint = {"decision": answer.decision, "how": answer.how}
                if answer.decision == "veto":
                    rnd.vetoed = True
                    proposer.reject_for_gain(record, "vetoed by the user", stage="vetoed")
            if not rnd.vetoed:
                await _decide(rnd, state, scorer, proposer, y, split, config)
        since_accept = 0 if rnd.accepted else since_accept + 1
        rnd.seconds = time.monotonic() - started
        state.rounds.append(rnd)
        await sink.round_done(rnd, state.champion)
        await hooks.emit("feature_decision", **_event(rnd, state.champion, tracker, cost))
    await hooks.check_cancelled()
    return await _finish(baseline, state, config, stop_reason, y, hooks, score=score_test)


async def _decide(
    rnd: Round,
    state: LoopState,
    scorer: FoldScorer,
    proposer: Proposer,
    y: np.ndarray,
    split: RowSplit,
    config: RunConfig,
) -> None:
    """Score champion + the new feature on the temporal folds and accept it by the rule."""
    record = rnd.record
    assert record.proposal is not None and record.values is not None
    name = record.proposal.name
    champion = state.champion
    candidate_frame = champion.frame.assign(**{name: record.values.to_numpy()})
    candidate_scores = await asyncio.to_thread(scorer.score, candidate_frame)
    rnd.gain = compare(champion.fold_scores, candidate_scores, config.rule, kind=scorer.kind)
    if not rnd.gain.accepted:
        proposer.reject_for_gain(
            record,
            f"no gain: {rnd.gain.mean_gain:+.4f} PR-AUC on the {scorer.kind} folds, "
            f"needs more than {rnd.gain.margin:.4f}",
        )
        return
    model, metrics, shares = await asyncio.to_thread(
        fit_validated, candidate_frame, y, split.train, split.val, config.seed
    )
    rnd.accepted = True
    rnd.champion_metrics = metrics
    state.champion = Champion(
        names=[*champion.names, name],
        frame=candidate_frame,
        metrics=metrics,
        shares=shares,
        best_iteration=int(metrics["best_iteration"]),
        fold_scores=candidate_scores,
        model=model,
    )
    proposer.adopt(record, shares.get(name, 0.0))


def _summary(record: ProposalRecord) -> dict[str, Any]:
    """What a person needs to approve a proposal: what it means, how it is computed, why."""
    p = record.proposal
    assert p is not None
    return {
        "name": p.name,
        "description": describe(p.ir) if p.ir is not None else p.rationale,
        "rationale": p.rationale,
        "expected_direction": p.expected_direction,
        "sql": record.sql,
        "ir": p.ir.model_dump(mode="json") if p.ir is not None else None,
    }


def _event(rnd: Round, champion: Champion, tracker: BudgetTracker, cost: float) -> dict[str, Any]:
    p = rnd.record.proposal
    out: dict[str, Any] = {
        "round": rnd.number,
        "name": p.name if p else None,
        "status": rnd.record.status,
        "stage": rnd.record.stage,
        "reasons": rnd.record.reasons[:3],
        "accepted": rnd.accepted,
        "vetoed": rnd.vetoed,
        "cost_usd": round(cost, 6),
        "champion_val_pr_auc": champion.metrics.get("pr_auc"),
        "budget_used": tracker.used(),
    }
    if rnd.gain is not None:
        out["gain"] = {
            "mean": rnd.gain.mean_gain,
            "ci95": list(rnd.gain.ci95),
            "margin": rnd.gain.margin,
            "metric": rnd.gain.metric,
        }
    return out


async def _finish(
    baseline: BaselineResult,
    state: LoopState,
    config: RunConfig,
    stop_reason: str,
    y: np.ndarray,
    hooks: LoopHooks,
    score: bool = True,
) -> RunOutcome:
    assert baseline.temporal is not None and state.tracker is not None
    split = baseline.temporal
    champion = state.champion
    status: RunStatus = "stopped" if stop_reason.startswith("budget") else "completed"
    test: dict[str, float] | None = None
    error: str | None = None
    if score:
        await hooks.step("Refitting on train + validation and scoring the test rows once", 0.95)
        test, error = await _score_once(champion, y, split, config.seed, baseline.report_threshold)
    await hooks.emit("run_finished", status=status, stop_reason=stop_reason, test=test, error=error)
    return RunOutcome(
        status, stop_reason, state.rounds, champion, test, error, state.tracker.used()
    )


async def _score_once(
    champion: Champion, y: np.ndarray, split: RowSplit, seed: int, with_threshold: bool = False
) -> tuple[dict[str, float] | None, str | None]:
    try:
        scored = await asyncio.to_thread(
            _refit_and_score_test, champion, y, split, seed, with_threshold
        )
        return scored, None
    except (SplitError, ValueError) as e:  # for example a test period with one class only
        return None, f"the test rows were not scored: {e}"


async def score_chosen(baseline: BaselineResult, outcome: RunOutcome, seed: int) -> RunOutcome:
    """Score the test rows for the one outcome that was kept among several rollouts.

    Rollouts run with ``score_test=False`` so the test rows are touched once per run, by the
    winner only, never once per rollout and never to choose between them.
    """
    assert baseline.labels is not None and baseline.temporal is not None
    y = baseline.labels["label"].astype(int).to_numpy()
    test, error = await _score_once(
        outcome.champion, y, baseline.temporal, seed, baseline.report_threshold
    )
    return replace(outcome, test=test, test_error=error)


def _refit_and_score_test(
    champion: Champion, y: np.ndarray, split: RowSplit, seed: int, with_threshold: bool = False
) -> dict[str, float]:
    rows = np.concatenate([split.train, split.val])
    if len(set(y[split.test])) < 2:
        raise ValueError("the test rows have only one class")
    params = {**LGBM_PARAMS, "n_estimators": max(10, champion.best_iteration)}
    model = lgb.LGBMClassifier(
        **params, random_state=seed, deterministic=True, force_row_wise=True, verbose=-1
    )
    model.fit(champion.frame.iloc[rows], y[rows])
    extra: dict[str, float] = {}
    if with_threshold:
        # chosen on the validation rows by the model that did not see them, as for any decision
        extra["threshold"] = validation_threshold(champion, y, split)
    return score_test(model, champion.frame.iloc[split.test], y[split.test], **extra)


def validation_threshold(champion: Champion, y: np.ndarray, split: RowSplit) -> float:
    """The probability cut-off with the best F1 on the validation rows (never the test rows)."""
    prob = np.asarray(champion.model.predict_proba(champion.frame.iloc[split.val]))[:, 1]
    return float(choose_threshold(y[split.val], prob)["value"])
