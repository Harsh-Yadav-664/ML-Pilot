"""The relational run loop (#58): proposals, acceptance by gain on temporal folds, the test once."""

from __future__ import annotations

# ruff: noqa: F811
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.agents import run_loop
from ml.agents.run_loop import LoopState, RunConfig
from ml.agents.run_loop import run_loop as execute
from ml.features.baseline import BaselineResult
from ml.features.engine import Budget
from tests.unit.test_dfs import demo, demo_baseline  # noqa: F401  (the demo world and its baseline)
from tests.unit.test_feature_engine import Costly, Recorder
from tests.unit.test_llm_sql import Scripted, answer, proposer

GOOD = ["good_refunds_14d", "good_cancelled_share", "good_ticket_count_60d"]
# margin = max(min_gain, std_multiplier * std): this makes it -1, so any gain above -1 is accepted
ACCEPT_ANY = {"min_gain": -1.0, "std_multiplier": -1e6}
ACCEPT_NONE = {"min_gain": 10.0}


class Cancelled(Exception):
    pass


class Hooks(Recorder):
    """Records events and steps; never cancels."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[str] = []

    async def step(self, name: str, progress: float | None = None, **payload: Any) -> None:
        self.steps.append(name)

    async def check_cancelled(self) -> None:
        return None


class CancelAfter(Hooks):
    """Hooks that report a cancel request from the n-th check on."""

    def __init__(self, checks: int) -> None:
        super().__init__()
        self.left = checks

    async def check_cancelled(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise Cancelled


class Sink:
    def __init__(self) -> None:
        self.rounds: list[tuple[int, bool, list[str]]] = []

    async def round_done(self, rnd: run_loop.Round, champion: run_loop.Champion) -> None:
        self.rounds.append((rnd.number, rnd.accepted, list(champion.names)))


@pytest.fixture
def test_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many rows each call of ``score_test`` was given: the spy on the test rows."""
    calls: list[int] = []
    real = run_loop.score_test

    def spy(model: Any, features: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        calls.append(len(y))
        return real(model, features, y)

    monkeypatch.setattr(run_loop, "score_test", spy)
    return calls


async def test_accepted_features_join_the_champion_and_the_test_rows_are_scored_once(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Costly([answer(k) for k in GOOD[:2]], cost=0.01)
    sink, events = Sink(), Hooks()
    p = proposer(demo, demo_baseline, gateway)
    out = await execute(
        demo_baseline,
        p,
        RunConfig(max_rounds=2, rule=ACCEPT_ANY),
        hooks=events,
        sink=sink,  # type: ignore[arg-type]
    )
    assert (out.status, out.stop_reason) == ("completed", "max_rounds")
    assert [r.accepted for r in out.rounds] == [True, True]
    assert out.champion.names[-2:] == ["refund_count_14d", "cancelled_order_share_180d"]
    assert len(out.champion.names) == len(demo_baseline.frame.columns) + 2  # type: ignore[union-attr]
    assert [r[1] for r in sink.rounds] == [True, True] and sink.rounds[-1][2] == out.champion.names
    assert len(test_calls) == 1 and out.test is not None and out.test["n_test"] == test_calls[0]
    assert test_calls[0] == len(demo_baseline.temporal.test)  # type: ignore[union-attr]
    assert out.used["proposals"] == 2 and out.used["cost_usd"] == pytest.approx(0.02)
    kinds = [k for k, _ in events.events]
    assert kinds.count("feature_decision") == 2 and kinds[-1] == "run_finished"
    decision = events.events[0][1]
    assert decision["accepted"] and decision["gain"]["metric"] == "pr_auc"
    # the next prompt lists the accepted feature among the features in use
    assert "refund" in gateway.prompts[1].text.lower()
    print(
        f"\nrun: {out.status} ({out.stop_reason}); {len(out.accepted)} of {len(out.rounds)} accepted; "
        f"test PR-AUC {out.test['pr_auc']:.4f} (base rate {out.test['base_rate']:.4f}), "
        f"score_test called {len(test_calls)} time(s) on {test_calls[0]} rows"
    )


async def test_a_feature_without_gain_is_rejected_in_code_and_the_next_prompt_says_why(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Scripted([answer(k) for k in GOOD])
    p = proposer(demo, demo_baseline, gateway)
    out = await execute(demo_baseline, p, RunConfig(max_rounds=3, patience=2, rule=ACCEPT_NONE))
    assert [r.accepted for r in out.rounds] == [False, False]  # stopped by patience after two
    assert out.stop_reason == "patience" and out.status == "completed"
    assert all(r.gain is not None and not r.gain.accepted for r in out.rounds)
    assert out.champion.names == [str(c) for c in demo_baseline.frame.columns]  # type: ignore[union-attr]
    assert "no gain" in gateway.prompts[1].text and "refund_count_14d" in gateway.prompts[1].text
    assert len(gateway.prompts) == 2
    assert len(test_calls) == 1


async def test_the_decision_is_the_rule_applied_to_paired_fold_scores(demo, demo_baseline) -> None:
    gateway = Scripted([answer("good_refunds_14d")])
    out = await execute(
        demo_baseline, proposer(demo, demo_baseline, gateway), RunConfig(max_rounds=1)
    )
    gain = out.rounds[0].gain
    assert gain is not None and len(gain.base_scores) == len(demo_baseline.temporal.folds)  # type: ignore[union-attr]
    assert gain.accepted == (gain.mean_gain > gain.margin) == out.rounds[0].accepted
    assert gain.rule["validation"] == "temporal"
    print(
        f"\ndefault rule on refund_count_14d: gain {gain.mean_gain:+.4f} "
        f"(95% CI {gain.ci95[0]:+.4f}..{gain.ci95[1]:+.4f}), margin {gain.margin:.4f} "
        f"-> {'accepted' if gain.accepted else 'rejected'}"
    )


async def test_a_leaky_proposal_never_reaches_the_gain_test(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Scripted([answer("leaky_sql"), answer("leaky_sql", name="again")])
    p = proposer(demo, demo_baseline, gateway, allow_free_sql=True)
    out = await execute(demo_baseline, p, RunConfig(max_rounds=1, rule=ACCEPT_ANY))
    rnd = out.rounds[0]
    assert rnd.record.status == "rejected_guard" and rnd.gain is None and not rnd.accepted
    assert out.champion.names == [str(c) for c in demo_baseline.frame.columns]  # type: ignore[union-attr]


async def test_a_budget_stops_the_run_and_the_champion_is_still_scored_once(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Costly([answer(k) for k in GOOD], cost=0.02)
    config = RunConfig(
        max_rounds=10, budget=Budget(max_cost_usd=0.05), rule=ACCEPT_NONE, patience=9
    )
    out = await execute(demo_baseline, proposer(demo, demo_baseline, gateway), config)
    assert (out.status, out.stop_reason) == ("stopped", "budget:cost")
    assert len(out.rounds) == 3 and out.used["cost_usd"] == pytest.approx(0.06)
    assert len(test_calls) == 1


async def test_without_a_real_model_the_baseline_is_the_result(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Scripted([answer("good_refunds_14d")], mode="fallback")
    out = await execute(demo_baseline, proposer(demo, demo_baseline, gateway), RunConfig())
    assert (out.status, out.stop_reason) == ("completed", "no_llm")
    assert out.rounds == [] and out.test is not None and len(test_calls) == 1
    assert out.champion.names == [str(c) for c in demo_baseline.frame.columns]  # type: ignore[union-attr]


async def test_cancelling_mid_run_keeps_the_last_champion_and_never_scores_the_test_rows(
    demo, demo_baseline, test_calls
) -> None:
    gateway = Costly([answer(k) for k in GOOD])
    p = proposer(demo, demo_baseline, gateway)
    sink = Sink()
    state, scorer = run_loop.start_state(demo_baseline, RunConfig(rule=ACCEPT_ANY))
    hooks = CancelAfter(checks=2)  # allows round 1 and round 2, cancels before round 3
    with pytest.raises(Cancelled):
        await execute(
            demo_baseline,
            p,
            RunConfig(max_rounds=3, rule=ACCEPT_ANY),
            hooks=hooks,  # type: ignore[arg-type]
            sink=sink,
            state=state,
            scorer=scorer,
        )
    assert len(state.rounds) == 2 and [r[0] for r in sink.rounds] == [1, 2]
    assert state.champion.names == sink.rounds[-1][2]  # what was last recorded is what stands
    assert len(state.champion.names) == len(demo_baseline.frame.columns) + 2  # type: ignore[union-attr]
    assert test_calls == []  # a cancelled run never touches the test rows


def test_the_folds_only_use_training_rows(demo_baseline: BaselineResult) -> None:
    split = demo_baseline.temporal
    assert split is not None and split.folds
    train = set(split.train.tolist())
    for fold in split.folds:
        rows = set(fold.train.tolist()) | set(fold.val.tolist())
        assert rows <= train
        assert not rows & set(split.test.tolist()) and not rows & set(split.val.tolist())
    assert dataclasses.is_dataclass(LoopState)
