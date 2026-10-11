"""The OpenML single-table benchmark (#67): the parts that do not need a download.

A small synthetic table stands in for an OpenML dataset; the splits, the metrics, the test-set
discipline, the result files and the table built from them are the real ones. The numbers of the
benchmark come from the workflow (.github/workflows/benchmark-openml.yml), never from here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import openml_harness as h
import openml_report as report
import openml_run as cli
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ai.gateway import PROVIDER_FACTORIES
from ml.agents import run_loop
from ml.features.table_baseline import build_table_baseline
from tests.unit.test_llm_sql import Scripted


def table(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """The target depends on the sum of four columns: a formula says it, trees find it slowly."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(1, 5)})
    df["plan"] = rng.choice(["basic", "plus", "pro"], size=n)
    signal = df[["x1", "x2", "x3", "x4"]].sum(axis=1) + 0.5 * rng.normal(size=n)
    df["churn"] = np.where(signal > 1.4, "yes", "no")
    return df


@pytest.fixture
def data() -> h.Data:
    frame, info = h.binary_target(table(), "churn")
    meta = {
        "source": "synthetic",
        "id": 0,
        "name": "toy",
        "licence": None,
        "target": "churn",
        **info,
    }
    return h.Data({**meta, "n_rows": len(frame), "n_features": frame.shape[1] - 1}, frame, "churn")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for setting, _, _ in PROVIDER_FACTORIES.values():
        monkeypatch.delenv(setting, raising=False)


class Scored:
    """Records what was scored: the benchmark's one test scoring, and the run loop's (never)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.loop_calls = 0
        self.fitted: set[int] = set()  # the rows any LightGBM fit saw, as training or eval data
        real_score, real_loop = h.score, run_loop.score_test

        def score(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
            self.calls.append((np.asarray(y), np.asarray(prob)))
            return real_score(y, prob)

        def loop_score(*args: Any, **kwargs: Any) -> dict[str, float]:
            self.loop_calls += 1
            return real_loop(*args, **kwargs)

        real_fit = lgb.LGBMClassifier.fit

        def fit(model: Any, X: pd.DataFrame, y: Any, **kwargs: Any) -> Any:
            self.fitted.update(X.index)
            for eval_x, _ in kwargs.get("eval_set") or []:
                self.fitted.update(eval_x.index)
            return real_fit(model, X, y, **kwargs)

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", fit)
        monkeypatch.setattr(h, "score", score)
        monkeypatch.setattr(run_loop, "score_test", loop_score)


def test_the_target_is_zero_one_with_the_rarer_class_as_one() -> None:
    frame, info = h.binary_target(table(), "churn")
    assert info["positive_class"] == "yes" and frame["churn"].isin([0, 1]).all()
    assert info["base_rate"] == round(float(frame["churn"].mean()), 4) < 0.5
    tie = pd.DataFrame({"y": ["a", "b"] * 5, "x": range(10)})
    assert h.binary_target(tie, "y")[1]["positive_class"] == "b"  # a tie: the later label
    with pytest.raises(h.BenchError, match="3 classes"):
        h.binary_target(pd.DataFrame({"y": list("abc")}), "y")
    with pytest.raises(h.BenchError, match="missing"):
        h.binary_target(pd.DataFrame({"y": ["a", None, "b"]}), "y")


def test_one_stratified_split_per_seed_and_the_baseline_uses_it(data: h.Data) -> None:
    train, val, test = h.split_rows(data.frame, data.target, seed=3)
    assert len({*train, *val, *test}) == len(data.frame) == len(train) + len(val) + len(test)
    assert [round(len(p) / len(data.frame), 2) for p in (train, val, test)] == [0.6, 0.2, 0.2]
    rates = [data.frame[data.target].iloc[p].mean() for p in (train, val, test)]
    assert max(rates) - min(rates) < 0.01  # stratified
    again = h.split_rows(data.frame, data.target, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip((train, val, test), again, strict=True))
    other = h.split_rows(data.frame, data.target, seed=4)
    assert h.split_hash(*other) != h.split_hash(train, val, test)
    # the product's own baseline splits the same rows, so every method shares the split
    _, task = build_table_baseline(data.frame, data.target, seed=3, positive_class=1)
    assert np.array_equal(task.split.test, test) and np.array_equal(task.split.train, train)


async def test_lightgbm_is_scored_once_and_the_test_labels_are_not_read_before(
    data: h.Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored = Scored(monkeypatch)
    record = await h.run_method(data, "lightgbm", h.Settings(seed=1))
    ((y, prob),) = scored.calls
    train, val, test = h.split_rows(data.frame, data.target, 1)
    assert len(y) == len(test) == record["rows"]["test"] and scored.loop_calls == 0
    assert record["test"]["roc_auc"] == roc_auc_score(y, prob)
    assert record["test"]["pr_auc"] == average_precision_score(y, prob)
    assert record["test"]["pr_auc"] > 2 * record["test"]["base_rate"]  # the toy table has signal
    assert record["llm"] is None and record["engine"]["name"] == "LightGBM"
    json.dumps(record)
    # no model was fitted or early-stopped on a test row, and every row of the other parts was used
    assert scored.fitted.isdisjoint(test) and scored.fitted == {*train, *val}


async def test_mlpilot_without_a_model_is_the_baseline_and_says_so(
    data: h.Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored = Scored(monkeypatch)
    profiled: list[int] = []  # what the model is told about the data comes from these rows
    real_profile = h.DataProfiler.profile

    def profile(self: Any, df: pd.DataFrame, **kwargs: Any) -> Any:
        profiled.append(len(df))
        return real_profile(self, df, **kwargs)

    monkeypatch.setattr(h.DataProfiler, "profile", profile)
    a = await h.run_method(data, "lightgbm", h.Settings(seed=2))
    b = await h.run_method(data, "mlpilot", h.Settings(seed=2))
    train, val, _ = h.split_rows(data.frame, data.target, 2)
    assert profiled == [len(train) + len(val)]  # the test rows' target stays out of the prompt
    assert b["llm"]["answered"] is False and b["llm"]["decision_modes"] == {"fallback": 1}
    assert b["llm"]["cost_usd_estimated"] == 0.0 or b["llm"]["answered_by"] == ["stub:stub-default"]
    assert b["agent"]["stop_reason"] == "no_llm" and b["agent"]["accepted"] == []
    assert b["test"] == a["test"] and b["split_hash"] == a["split_hash"]
    assert scored.loop_calls == 0 and len(scored.calls) == 2  # one scoring per run


async def test_mlpilot_with_a_model_keeps_a_feature_only_when_the_rule_accepts_it(
    data: h.Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored = Scored(monkeypatch)
    answers = [
        {"name": "total", "formula": "x1 + x2 + x3 + x4", "reason": "the sum is the signal"},
        {"name": "x3_doubled", "formula": "x3 + x3", "reason": "a rescaled copy"},
    ]
    settings = h.Settings(seed=0, max_rounds=2, patience=2, model="fake:model")
    record = await h.run_method(data, "mlpilot", settings, gateway=Scripted(answers))
    assert record["llm"]["answered"] is True and record["llm"]["answered_by"] == ["fake:fake-1"]
    assert record["llm"]["decision_modes"] == {"llm": 2}
    assert record["agent"]["accepted"] == ["total"]  # the copy adds nothing and is rejected
    assert record["features"]["n"] == 6 and scored.loop_calls == 0 and len(scored.calls) == 1
    _, _, test = h.split_rows(data.frame, data.target, 0)
    assert scored.fitted.isdisjoint(test)  # candidates were judged on folds of train + validation
    base = await h.run_method(data, "lightgbm", h.Settings(seed=0))
    assert record["test"]["pr_auc"] > base["test"]["pr_auc"]


def _record(data: h.Data, method: str, seed: int, pr: float, roc: float, seconds: float) -> dict:
    train, val, test = h.split_rows(data.frame, data.target, seed)
    return {
        "schema_version": h.SCHEMA_VERSION,
        "status": "ok",
        "dataset": data.meta,
        "method": method,
        "seed": seed,
        "test": {"pr_auc": pr, "roc_auc": roc, "base_rate": 0.3, "n": 240.0},
        "split_hash": h.split_hash(train, val, test),
        "engine": {
            "name": "AutoGluon",
            "version": "x",
            "preset": "p",
            "time_limit_seconds": 9,
            "eval_metric": "roc_auc",
        },
        "llm": {"answered": False} if method == "mlpilot" else None,
        "agent": {"accepted": [], "stop_reason": "no_llm"} if method == "mlpilot" else None,
        "wall_seconds": seconds,
        "environment": h.environment(),
    }


def test_the_table_is_built_from_the_result_files_only(data: h.Data, tmp_path: Path) -> None:
    pr = {
        "lightgbm": [0.70, 0.72, 0.74],
        "mlpilot": [0.71, 0.73, 0.75],
        "autogluon": [0.80, 0.82, 0.84],
    }
    for method, values in pr.items():
        for seed, value in enumerate(values):
            record = _record(data, method, seed, value, value + 0.1, 5.0)
            h.write_result(tmp_path, record)
    failed = _record(data, "autogluon", 3, 0, 0, 0) | {"status": "failed", "error": "boom"}
    h.write_result(tmp_path, failed)
    text = report.write(tmp_path, tmp_path / "RESULTS.md")
    assert (tmp_path / "RESULTS.md").read_text() == text
    assert "0.7200 ± 0.0200" in text and "**0.8200 ± 0.0200** (1 failed)" in text  # best is bold
    assert "0.9200 ± 0.0200" in text  # the ROC-AUC table, from the same files
    assert "MLPilot minus AutoGluon" in text and "-0.0900 ± 0.0000" in text
    assert "MLPilot higher on 0 of 1 datasets" in text  # honest: AutoGluon wins this one
    assert "none of the 3 mlpilot runs was answered by a model" in text
    assert "same train / validation / test rows" in text
    # every number in the table is one of the files' numbers or computed from them
    assert not re.search(r"DIFFERENT SPLITS", text)


def test_a_method_that_used_another_split_is_flagged(data: h.Data, tmp_path: Path) -> None:
    h.write_result(tmp_path, _record(data, "lightgbm", 0, 0.7, 0.8, 1))
    other = _record(data, "mlpilot", 0, 0.7, 0.8, 1) | {"split_hash": "different"}
    h.write_result(tmp_path, other)
    assert "DIFFERENT SPLITS across methods for toy (0)" in report.render(report.collect(tmp_path))


def test_the_command_line_writes_a_result_and_the_table(data: h.Data, tmp_path: Path) -> None:
    csv = tmp_path / "toy.csv"
    table().to_csv(csv, index=False)
    out = tmp_path / "results"
    for seed in ("0", "1"):
        args = ["--csv", str(csv), "--target", "churn", "--method", "lightgbm", "--seed", seed]
        assert cli.main([*args, "--results-dir", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*/*.json")) == [
        "lightgbm_seed0.json",
        "lightgbm_seed1.json",
    ]
    md = tmp_path / "RESULTS.md"
    assert cli.main(["--aggregate", "--results-dir", str(out), "--markdown", md.as_posix()]) == 0
    assert "## Test PR-AUC" in md.read_text() and "±" in md.read_text()
    with pytest.raises(SystemExit):
        cli.parse(["--method", "mlpilot"])  # a dataset is needed
    assert h.not_run(data, "autogluon", h.Settings(), "failed", error="x")["status"] == "failed"


def test_openml_loading_reads_the_licence_and_makes_the_target_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openml = pytest.importorskip(
        "openml", reason="pip install -r benchmarks/openml/requirements.txt"
    )
    X = table().drop(columns="churn")
    y = table()["churn"]
    fake = SimpleNamespace(
        name="toy", version=1, licence="CC0", default_target_attribute="churn",
        get_data=lambda target: (X, y, [False] * 5, list(X.columns)),
    )  # fmt: skip
    seen: dict[str, Any] = {}

    def get_dataset(dataset_id: int, **kwargs: Any) -> SimpleNamespace:
        seen.update(id=dataset_id, **kwargs)
        return fake

    monkeypatch.setattr(openml.datasets, "get_dataset", get_dataset)
    loaded = h.load_openml(31)
    assert seen["id"] == 31 and loaded.meta["licence"] == "CC0" and loaded.meta["name"] == "toy"
    assert loaded.meta["url"].endswith("/d/31") and loaded.frame["churn"].isin([0, 1]).all()
    assert set(h.DATASETS) >= {31, 37} and len(h.DATASETS) == 10


def test_autogluon_gets_the_train_and_validation_rows_and_never_the_test_rows(
    data: h.Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With AutoGluon installed this runs it for real (a few seconds); the split is checked either way."""
    seen: dict[str, Any] = {}

    def fake(frame: pd.DataFrame, target: str, fit: np.ndarray, test: np.ndarray, s: h.Settings):
        seen.update(fit=fit, test=test)
        prob = np.linspace(0, 1, len(test))
        return prob, {"name": "AutoGluon", "version": "fake"}

    monkeypatch.setattr(h, "ag_scores", fake)
    import asyncio

    record = asyncio.run(h.run_method(data, "autogluon", h.Settings(seed=5)))
    train, val, test = h.split_rows(data.frame, data.target, 5)
    assert np.array_equal(seen["test"], test) and not set(seen["fit"]) & set(test)
    assert set(seen["fit"]) == {*train, *val} and record["method"] == "autogluon"


def test_autogluon_runs_on_the_same_split(data: h.Data) -> None:
    pytest.importorskip(
        "autogluon.tabular", reason="pip install -r benchmarks/openml/requirements.txt"
    )
    import asyncio

    settings = h.Settings(seed=0, ag_seconds=30, ag_preset="medium_quality")
    ag = asyncio.run(h.run_method(data, "autogluon", settings))
    lgbm = asyncio.run(h.run_method(data, "lightgbm", settings))
    assert ag["split_hash"] == lgbm["split_hash"] and ag["test"]["n"] == lgbm["test"]["n"]
    assert ag["test"]["roc_auc"] > 0.7 and ag["engine"]["time_limit_seconds"] == 30


def test_the_openml_datasets_are_pinned_and_distinct() -> None:
    assert len(set(h.DATASETS.values())) == len(h.DATASETS) == 10
