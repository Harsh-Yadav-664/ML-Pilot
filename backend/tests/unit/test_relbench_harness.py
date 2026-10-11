"""The RelBench harness (#65) on a tiny dataset in RelBench's own format.

The data is synthetic (tests/fixtures/relbench_toy.py) but everything around it is real: the
dataset is loaded by ``relbench.load_dataset``, its label tables and timestamps are RelBench's, and
the numbers come out of ``task.evaluate``. These tests prove the plumbing; the rel-f1 numbers come
from the benchmark workflow (.github/workflows/benchmark.yml), never from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("relbench", reason="pip install -r benchmarks/requirements.txt")

import relbench_harness as h
import relbench_report as report
import run as bench_cli

from tests.fixtures import relbench_toy
from tests.unit.test_llm_sql import Scripted


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return relbench_toy.build(tmp_path_factory.mktemp("toy-f1"))


@pytest.fixture
def bench(toy: Path) -> h.Bench:
    dataset, task = h.load_task(str(toy), relbench_toy.TASK)
    return h.translate(dataset, task)


class Spy:
    """Records the calls on a RelBench task that could read its test labels."""

    def __init__(self, task: Any) -> None:
        self.test_reads, self.evaluations = 0, []
        get_table, evaluate = task.get_table, task.evaluate

        def watched_get_table(
            split: str, mask_input_cols: bool | None = None, db: Any = None
        ) -> Any:
            if split == "test" and mask_input_cols is False:
                self.test_reads += 1
            return get_table(split, mask_input_cols=mask_input_cols, db=db)

        def watched_evaluate(pred: np.ndarray, target_table: Any = None) -> Any:
            out = evaluate(pred, target_table)
            self.evaluations.append((np.asarray(pred), target_table is None, out))
            return out

        task.get_table, task.evaluate = watched_get_table, watched_evaluate


def test_the_task_is_mapped_with_relbenchs_labels_and_timestamps(bench: h.Bench) -> None:
    task = bench.task
    train, val = task.get_table("train").df, task.get_table("val").df
    test = task.get_table("test").df
    assert bench.rows == {"train": len(train), "val": len(val), "test": len(test)}
    assert len(bench.labels) == len(train) + len(val) + len(test)
    # RelBench's val and test timestamps are the split, to the second
    assert bench.plan.val_from.replace(tzinfo=None) == pd.Timestamp(relbench_toy.VAL_TIMESTAMP)
    assert bench.plan.test_from.replace(tzinfo=None) == pd.Timestamp(relbench_toy.TEST_TIMESTAMP)
    assert bench.spec.horizon == "30d" and bench.spec.entity.table == "drivers"
    # the labels are RelBench's: same rows, same values; the test labels are not handed over
    got = bench.labels.iloc[: len(train)]
    assert got["label"].to_numpy().tolist() == train["did_not_finish"].tolist()
    assert (bench.labels["label"].iloc[len(train) + len(val) :] == 0).all()
    assert (bench.labels["label_window_end"] - bench.labels["cutoff_time"] == task.timedelta).all()
    # keys and time columns are RelBench's, tables without a time column are static
    tables = {t.key: t for t in bench.graph.tables}
    assert tables["results"].time_column == "date" and tables["drivers"].time_column is None
    assert tables["drivers"].is_static is True and tables["drivers"].primary_key == ["driverId"]
    edges = {(e.from_table, e.from_columns[0], e.to_table) for e in bench.graph.edges}
    assert edges == {("results", "raceId", "races"), ("results", "driverId", "drivers")}


def test_a_task_that_is_not_binary_is_refused(toy: Path) -> None:
    from relbench.base import TaskType

    dataset, task = h.load_task(str(toy), relbench_toy.TASK)
    task.task_type = TaskType.REGRESSION
    with pytest.raises(h.BenchError, match="binary tasks only"):
        h.translate(dataset, task)


async def test_method_b_is_scored_once_by_relbenchs_evaluator(bench: h.Bench) -> None:
    spy = Spy(bench.task)
    record = await h.run_method(bench, "b", h.Settings(seed=1))
    # the evaluator ran on the test table once (and on the validation table once)
    assert spy.test_reads == 1
    (val_call, test_call) = spy.evaluations
    assert val_call[1] is False and test_call[1] is True
    pred, _, out = test_call
    truth = bench.task.get_table("test", mask_input_cols=False).df["did_not_finish"].to_numpy()
    assert len(pred) == len(truth) == bench.rows["test"]
    assert record["test"]["roc_auc"] == out["roc_auc"] == roc_auc_score(truth, pred)
    assert record["status"] == "ok" and record["llm"] is None
    assert record["test"]["roc_auc"] > 0.55  # the toy history carries signal
    assert record["rows"]["split"]["n_purged_train"] == 0
    assert record["features"]["n"] > 5
    json.dumps(record)


async def test_method_a_uses_the_entity_table_columns_only(bench: h.Bench) -> None:
    a = await h.run_method(bench, "a", h.Settings(seed=0))
    b = await h.run_method(bench, "b", h.Settings(seed=0))
    assert set(a["features"]["names"]) <= {"drivers__style", "drivers__nationality"}
    assert a["features"]["n"] < b["features"]["n"]


async def test_test_labels_do_not_change_what_mlpilot_predicts(bench: h.Bench) -> None:
    first = await h.run_method(bench, "b", h.Settings(seed=2))
    flipped = bench.labels.copy()
    test_rows = flipped["cutoff_time"] >= bench.plan.test_from.replace(tzinfo=None)
    flipped.loc[test_rows, "label"] = 1
    again = await h.run_method(
        h.Bench(**{**bench.__dict__, "labels": flipped}), "b", h.Settings(seed=2)
    )
    assert again["test"] == first["test"]


async def test_the_agent_without_a_model_is_the_baseline_and_says_so(
    bench: h.Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    for setting, _, _ in h.PROVIDER_FACTORIES.values():
        monkeypatch.delenv(setting, raising=False)
    c = await h.run_method(bench, "c", h.Settings(seed=3))
    b = await h.run_method(bench, "b", h.Settings(seed=3))
    assert c["llm"]["answered"] is False and c["llm"]["decision_modes"] == {"fallback": 1}
    assert c["llm"]["answered_by"] == ["stub:stub-default"]
    assert c["agent"]["stop_reason"] == "no_llm" and c["agent"]["accepted"] == []
    assert c["test"] == b["test"]
    assert c["budget"] == {
        "max_rounds": 10,
        "patience": 5,
        "max_cost_usd": 1.0,
        "max_seconds": 1800.0,
    }


async def test_the_agents_model_cost_and_tokens_are_recorded(bench: h.Bench) -> None:
    ir = {
        "name": "dnf_14d",
        "entity_table": "drivers",
        "path": [
            {
                "from_table": "results",
                "from_columns": ["driverId"],
                "to_table": "drivers",
                "to_columns": ["driverId"],
            }
        ],
        "source_table": "results",
        "agg": "sum",
        "column": "dnf",
        "window_days": 14,
    }
    proposal = {"name": "dnf_14d", "rationale": "recent retirements", "ir": ir}
    gateway = Scripted([proposal])
    record = await h.run_method(
        bench, "c", h.Settings(seed=0, max_rounds=1, model="anthropic:fake"), gateway=gateway
    )
    assert record["llm"]["answered"] is True and record["llm"]["calls"] == 1
    assert record["llm"]["answered_by"] == ["fake:fake-1"]
    assert record["llm"]["requested_model"] == "anthropic:fake"
    assert "cost_usd_estimated" in record["llm"] and "tokens_in_estimated" in record["llm"]
    (rnd,) = record["agent"]["rounds"]
    assert rnd["name"] == "dnf_14d" and rnd["status"] == "proposed"
    assert record["wall_seconds"]["total"] > 0


def test_method_d_is_skipped_without_a_model_and_a_key(
    toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(h.ENV_STRONG_MODEL, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["--dataset", str(toy), "--task", relbench_toy.TASK, "--method", "d"]
    assert bench_cli.main([*args, "--results-dir", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "toy-f1" / relbench_toy.TASK / "d_seed0.json").read_text())
    assert record["status"] == "skipped" and "test" not in record
    assert h.ENV_STRONG_MODEL in record["reason"]
    # a model whose key is missing is skipped too, never run on the stub under the strong label
    monkeypatch.setenv(h.ENV_STRONG_MODEL, "anthropic:claude-x")
    assert not h.llm_ready("anthropic:claude-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    assert h.llm_ready("anthropic:claude-x")


def test_the_cli_writes_one_json_per_run_and_the_table_is_computed_from_them(
    toy: Path, tmp_path: Path
) -> None:
    for seed in (0, 1):
        code = bench_cli.main(
            ["--dataset", str(toy), "--task", relbench_toy.TASK, "--method", "b"]
            + ["--seed", str(seed), "--results-dir", str(tmp_path)]
        )
        assert code == 0
    files = sorted(tmp_path.glob("*/*/*.json"))
    assert [f.name for f in files] == ["b_seed0.json", "b_seed1.json"]
    records = [json.loads(f.read_text()) for f in files]
    md = tmp_path / "RESULTS.md"
    bench_cli.main(["--aggregate", "--results-dir", str(tmp_path), "--markdown", str(md)])
    scores = [r["test"]["roc_auc"] for r in records]
    assert report.mean_std(scores) in md.read_text()
    assert "toy-f1 / driver-dnf" in md.read_text()


def test_mean_and_std_over_seeds() -> None:
    assert report.mean_std([0.70, 0.74, 0.72]) == "0.7200 ± 0.0200"
    assert report.mean_std([0.7]) == "0.7000 (1 seed)"


def test_a_method_without_numbers_is_shown_as_skipped_with_its_reason() -> None:
    env = {"relbench": "x", "lightgbm": "y", "python": "z", "mlpilot_commit": None}
    ok = {
        "status": "ok",
        "dataset": "d",
        "task": "t",
        "method": "b",
        "seed": 0,
        "test": {"roc_auc": 0.7},
        "val": {"roc_auc": 0.8},
        "wall_seconds": {"total": 3.0},
        "llm": None,
        "environment": env,
    }
    skip = {**ok, "status": "skipped", "method": "d", "reason": "no strong model"}
    text = report.render([ok, skip])
    assert "| b: DFS features + LightGBM | 0.7000 (1 seed) | 0.8000 (1 seed) | 1 |" in text
    assert "| d: MLPilot agent, strong model | skipped: no strong model |" in text
