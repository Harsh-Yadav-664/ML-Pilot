"""A single-table task on the one run loop (#151): baseline, formulas, the paired rule, the test once."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ml.agents import run_loop
from ml.agents.hooks import NoHooks
from ml.agents.run_loop import RunConfig
from ml.data.profiling.profiler import DataProfiler
from ml.experiments import acceptance
from ml.features import gain
from ml.features.baseline import BaselineError, BaselineResult
from ml.features.formula_proposer import FormulaProposer
from ml.features.table_baseline import TableTask, build_table_baseline
from ml.validation.splits import SplitError

BACKEND = Path(__file__).resolve().parents[2]


def table(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Churn depends on the sum of four columns: trees find that slowly, one formula says it."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],  # id-like: dropped by the preparation
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "x4": rng.normal(size=n),
            "plan": rng.choice(["basic", "plus", "pro"], size=n),
            "spend": [str(v) for v in rng.uniform(10, 90, size=n).round(2)],  # numbers as text
        }
    )
    signal = df["x1"] + df["x2"] + df["x3"] + df["x4"] + 0.5 * rng.normal(size=n)
    df["churn"] = np.where(signal > 1.6, "Yes", "No")
    return df


class Planner:
    """Stands in for the model behind ``ExperimentPlanner``: answers from a script."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    async def generate_next_hypothesis(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.answers.pop(0)


def answer(name: str, formula: str) -> dict[str, Any]:
    return {"name": name, "formula": formula, "reason": f"try {name}"}


def start(df: pd.DataFrame, answers: list[dict[str, Any]]) -> tuple[BaselineResult, Any, Planner]:
    baseline, task = build_table_baseline(df, "churn")
    planner = Planner(answers)
    profile = DataProfiler().profile(df, target_column="churn")
    return (
        baseline,
        FormulaProposer(
            planner=planner,  # type: ignore[arg-type]
            profile=profile,
            task=task,
            baseline=baseline,
        ),
        planner,
    )


@pytest.fixture
def test_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many rows each call of ``score_test`` was given: the spy on the test rows."""
    calls: list[int] = []
    real = run_loop.score_test

    def spy(model: Any, features: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        calls.append(len(y))
        return real(model, features, y, **kwargs)

    monkeypatch.setattr(run_loop, "score_test", spy)
    return calls


async def test_a_csv_run_scores_the_test_rows_once_and_accepts_by_the_paired_rule(
    test_calls: list[int],
) -> None:
    df = table()
    baseline, proposer, planner = start(
        df, [answer("total", "x1 + x2 + x3 + x4"), answer("x3_doubled", "x3 + x3")]
    )
    out = await run_loop.run_loop(baseline, proposer, RunConfig(max_rounds=2), hooks=NoHooks())

    assert (out.status, out.stop_reason) == ("completed", "max_rounds")
    assert [r.record.proposal.name for r in out.rounds if r.record.proposal] == [
        "total",
        "x3_doubled",
    ]
    # the planted sum is a real gain; a rescaled copy of a column adds nothing to a tree model
    assert [r.accepted for r in out.rounds] == [True, False]
    assert out.champion.names[-1] == "total"
    first = out.rounds[0].gain
    assert first is not None and first.accepted and first.mean_gain > first.margin
    assert first.rule["validation"] == "random" and len(first.base_scores) == 15

    # the test rows: scored once, by the loop's score_test, with every decision already made
    assert test_calls == [len(baseline.temporal.test)]  # type: ignore[union-attr]
    assert out.test is not None and out.test["n_test"] == test_calls[0]
    assert {"pr_auc", "base_rate", "f1", "accuracy", "precision", "recall", "threshold"} <= set(
        out.test
    )
    assert out.test["pr_auc"] > out.test["base_rate"]
    # the planner was told what the loop decided
    history = planner.calls[1]["history"]
    assert history[-1]["name"] == "total" and history[-1]["decision"] == "keep"
    print(
        f"\nCSV run: {len(out.accepted)} of {len(out.rounds)} accepted; first gain "
        f"{first.mean_gain:+.4f} PR-AUC (margin {first.margin:.4f}); test PR-AUC "
        f"{out.test['pr_auc']:.4f} (base rate {out.test['base_rate']:.4f}); "
        f"score_test called {len(test_calls)} time(s) on {test_calls[0]} rows"
    )


async def test_the_test_rows_are_in_no_fit_and_in_one_prediction_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spy on the model: no fit and no fold ever sees a test row; one predict_proba does."""
    df = table(1200)
    baseline, proposer, _ = start(df, [answer("product", "x1 * x2"), answer("p2", "x1 * x3")])
    test_rows = set(baseline.temporal.test.tolist())  # type: ignore[union-attr]
    fit_rows: list[set[int]] = []
    predict_rows: list[set[int]] = []
    real_fit, real_predict = lgb.LGBMClassifier.fit, lgb.LGBMClassifier.predict_proba

    def fit(self: Any, X: pd.DataFrame, y: Any, *a: Any, **k: Any) -> Any:
        fit_rows.append(set(X.index.tolist()))
        eval_set = k.get("eval_set")
        if eval_set:
            fit_rows.extend(set(e[0].index.tolist()) for e in eval_set)
        return real_fit(self, X, y, *a, **k)

    def predict_proba(self: Any, X: pd.DataFrame, *a: Any, **k: Any) -> Any:
        predict_rows.append(set(X.index.tolist()))
        return real_predict(self, X, *a, **k)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", fit)
    monkeypatch.setattr(lgb.LGBMClassifier, "predict_proba", predict_proba)
    out = await run_loop.run_loop(baseline, proposer, RunConfig(max_rounds=2), hooks=NoHooks())

    assert out.test is not None
    assert fit_rows and all(not (rows & test_rows) for rows in fit_rows)
    touching = [rows for rows in predict_rows if rows & test_rows]
    assert len(touching) == 1 and touching[0] == test_rows
    # and every acceptance fold is drawn from the train and validation rows
    allowed = set(baseline.temporal.fold_rows.tolist())  # type: ignore[union-attr]
    assert not (allowed & test_rows)
    for fold in baseline.temporal.folds:  # type: ignore[union-attr]
        assert set(fold.train.tolist()) | set(fold.val.tolist()) <= allowed


async def test_a_cancelled_csv_run_is_never_scored(test_calls: list[int]) -> None:
    class Cancel(NoHooks):
        async def check_cancelled(self) -> None:
            raise RuntimeError("cancelled")

    baseline, proposer, _ = start(table(800), [answer("product", "x1 * x2")])
    with pytest.raises(RuntimeError, match="cancelled"):
        await run_loop.run_loop(baseline, proposer, RunConfig(max_rounds=1), hooks=Cancel())
    assert test_calls == []


async def test_without_a_real_model_nothing_is_proposed_and_the_baseline_stands(
    test_calls: list[int],
) -> None:
    offline = {**answer("x", "x1 * x2"), "llm": {"decision_mode": "fallback"}}
    baseline, proposer, _ = start(table(800), [offline])
    out = await run_loop.run_loop(baseline, proposer, RunConfig(max_rounds=3), hooks=NoHooks())
    assert out.stop_reason == "no_llm" and out.test is not None
    assert out.champion.names == [str(c) for c in baseline.frame.columns]  # type: ignore[union-attr]
    assert len(test_calls) == 1


def test_a_formula_is_data_and_goes_through_the_safe_evaluator() -> None:
    df = table(800)
    baseline, proposer, _ = start(df, [])

    async def ask(name: str, formula: str) -> Any:
        proposer.planner.answers.append(answer(name, formula))  # type: ignore[attr-defined]
        return await proposer.propose()

    import asyncio

    cases = {
        "evil": ("__import__('os').getcwd()", "guard"),
        "attr": ("().__class__", "guard"),
        "ghost": ("no_such_column * 2", "guard"),
        "const": ("x1 * 0 + 1", "execution"),
        "text": ("plan * 2", "execution"),
        "x1": ("x2 + 1", "duplicate"),  # the name is a column already
        "copy": ("x1 + 0", "duplicate_after_execution"),
    }
    for name, (formula, stage) in cases.items():
        record = asyncio.run(ask(name, formula))
        assert record.status != "proposed" and record.stage == stage, (name, record)
    # the same formula is not tried twice, whatever the spacing
    ok = asyncio.run(ask("good", "x1*x2"))
    assert ok.status == "proposed" and ok.values is not None
    again = asyncio.run(ask("good2", "x1 *  x2"))
    assert again.status == "rejected_duplicate"
    # nothing was ever evaluated outside the whitelist: no rejected record holds a value
    assert all(r.values is None for r in proposer.records if r.status != "proposed")
    assert baseline.frame is not None


async def test_a_feature_adopted_by_the_champion_can_be_used_by_the_next_formula() -> None:
    _, proposer, _ = start(
        table(800), [answer("product", "x1 * x2"), answer("product_sq", "product * product")]
    )
    first = await proposer.propose()
    assert first.status == "proposed"
    proposer.adopt(first, 0.1)
    second = await proposer.propose()
    assert second.status == "proposed"  # 'product' is a column now


def test_the_baseline_is_the_existing_preprocessing_and_a_random_split() -> None:
    df = table(1000)
    baseline, task = build_table_baseline(df, "churn")
    assert isinstance(task, TableTask)
    assert baseline.frame is not None
    assert "customer_id" not in baseline.frame.columns  # id-like, dropped as before
    assert "customer_id" in task.report["excluded_features"]
    assert str(baseline.frame["spend"].dtype) == "float64"  # numbers stored as text, converted
    assert str(baseline.frame["plan"].dtype) == "category"
    assert task.encoder.classes == ["No", "Yes"] and task.encoder.positive_class == "Yes"
    split = baseline.temporal
    assert split is not None
    parts = [set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sum(len(p) for p in parts) == len(df)
    assert baseline.fold_kind == "random" and baseline.report_threshold
    assert baseline.metrics["pr_auc"] > baseline.metrics["base_rate"]


def test_a_target_the_loop_cannot_handle_fails_loudly() -> None:
    df = table(600)
    with pytest.raises(BaselineError, match="two-class"):
        build_table_baseline(df.assign(churn=np.arange(len(df)) % 3), "churn")
    with pytest.raises(BaselineError, match="not found"):
        build_table_baseline(df, "nope")
    with pytest.raises(BaselineError, match="missing values"):
        build_table_baseline(df.assign(churn=df["churn"].where(df.index > 5)), "churn")
    with pytest.raises((BaselineError, SplitError)):
        build_table_baseline(df.assign(churn="No"), "churn")


def test_there_is_one_acceptance_function() -> None:
    """Both validation schemes decide with ``acceptance.decide``; no other code applies the rule."""
    base, cand = [0.50, 0.52, 0.51, 0.49, 0.50], [0.55, 0.56, 0.54, 0.53, 0.55]
    direct = acceptance.decide(base, cand, None, metric="pr_auc")
    via_loop = gain.compare(base, cand, None, kind="random")
    for field in ("mean_gain", "std_gain", "ci95", "margin", "accepted", "metric"):
        assert getattr(direct, field) == getattr(via_loop, field)
    rule_sources = [
        p.relative_to(BACKEND).as_posix()
        for folder in ("app", "ml")
        for p in (BACKEND / folder).rglob("*.py")
        if re.search(r"mean_gain\s*>\s*margin", p.read_text())
    ]
    assert rule_sources == ["ml/experiments/acceptance.py"]


def test_the_old_formula_loop_is_gone() -> None:
    assert importlib.util.find_spec("ml.agents.decision_agent") is None
    leftovers = [
        p.relative_to(BACKEND).as_posix()
        for folder in ("app", "ml", "tests")
        for p in (BACKEND / folder).rglob("*.py")
        if re.search(r"DecisionAgent|run_optimization_loop", p.read_text())
        and p.name != "test_table_run_loop.py"
    ]
    assert leftovers == []
