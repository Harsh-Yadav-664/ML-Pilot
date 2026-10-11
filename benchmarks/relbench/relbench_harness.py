"""MLPilot on a RelBench task, scored by RelBench's own evaluator (issue #65).

RelBench ships the database, the label tables of each task (train, val, test), the val and test
timestamps and the metric. This module maps one binary task onto MLPilot and changes nothing
about it:

* the **labels are RelBench's** (``task.get_table``); MLPilot does not re-derive them. A label row is
  (entity, cutoff = the row's timestamp, label), its label window ends one ``timedelta`` later;
* the **split is RelBench's**: train, val and test are told apart by the dataset's val and test
  timestamps (``TemporalSplitPlan``), and the database MLPilot sees is ``task.get_db()``: rows up
  to the test timestamp, the task's hidden columns removed. Features still pass the point-in-time
  guard, so a row only reads data from before its own cutoff;
* the **test labels are not given to MLPilot** (the frame it gets holds 0 for test rows, and is
  never used for them). They are read once, by ``task.evaluate``, at the very end;
* the **metric is RelBench's** (``task.evaluate``: ROC-AUC for a binary task).

Methods (the same budget, recorded, for the two that use a language model):

    a  LightGBM on the entity table's own columns (no history; ids, timestamps and
       high-cardinality text are not used)
    b  DFS features + LightGBM: the MLPilot baseline (#55), no LLM
    c  MLPilot agent (#58) with a cheap model
    d  MLPilot agent with a strong model; skipped, and labelled so, without a model and its key

Without an LLM key the agent's gateway is the offline stub, no feature is proposed, and c is b run
through the loop: the result says so (``llm.answered`` is false). Nothing here invents a number: a
result file is written from a run, and ``relbench_report`` only reads result files.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from ai.context_builder import ContextBuilder
from ai.gateway import PROVIDER_FACTORIES, AIGateway
from ai.llm_config import parse_entry, validate_tiers
from ai.router import TaskRouter
from ml.agents import run_loop as loop
from ml.data.schema_graph import Column, Edge, SchemaGraph, Table, column_hint, type_kind
from ml.features.baseline import BaselineResult, build_baseline, flatten_graph, typed_times
from ml.features.engine import Budget, FeatureEngine
from ml.features.llm_sql import FeatureProposer
from ml.tasks.pit_guard import CUTOFF, ENTITY
from ml.tasks.spec import TaskSpec, from_dict
from ml.validation.splits import TemporalSplitPlan

HUB = "stanford-star/relbench-v1"
SCHEMA_VERSION = 1
METHODS = {
    "a": "LightGBM on the entity table's columns",
    "b": "DFS features + LightGBM",
    "c": "MLPilot agent, cheap model",
    "d": "MLPilot agent, strong model",
}
ALIASES = {"mlpilot": "c"}
ENV_CHEAP_MODEL = "MLPILOT_BENCH_CHEAP_MODEL"
ENV_STRONG_MODEL = "MLPILOT_BENCH_STRONG_MODEL"
REPO = Path(__file__).resolve().parents[2]


class BenchError(ValueError):
    """The task cannot be run as asked (never a silent partial result)."""


@dataclass(frozen=True)
class Settings:
    """What one run is allowed to spend. The same for every task and for c and d."""

    seed: int = 0
    max_rounds: int = 10
    patience: int = 5
    max_cost_usd: float | None = 1.0
    max_seconds: float | None = 1800.0
    model: str | None = None  # "provider:model" for the SQL tier, c and d only

    def budget(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "patience": self.patience,
            "max_cost_usd": self.max_cost_usd,
            "max_seconds": self.max_seconds,
        }


@dataclass
class Bench:
    """One RelBench task, translated: everything MLPilot needs, and the task for the evaluator."""

    dataset: str
    task: Any  # relbench EntityTask
    spec: TaskSpec
    graph: SchemaGraph
    tables: dict[str, pa.Table]
    labels: pd.DataFrame  # entity_id, cutoff_time, label, label_window_end; test labels are 0
    plan: TemporalSplitPlan
    rows: dict[str, int]  # RelBench's label rows per split


# -- loading and translating ---------------------------------------------------------------


def load_task(dataset: str, task_name: str) -> tuple[Any, Any]:
    """The RelBench dataset and task. A bare name such as ``rel-f1`` is fetched from the hub."""
    import relbench

    ref = dataset if "/" in dataset or Path(dataset).exists() else f"{HUB}/{dataset}"
    ds = relbench.load_dataset(ref)
    return ds, ds.load_task(task_name)


def translate(dataset: Any, task: Any) -> Bench:
    """Map a RelBench binary task to an MLPilot task spec, graph, tables and label rows."""
    from relbench.base import TaskType

    if task.task_type != TaskType.BINARY_CLASSIFICATION:
        raise BenchError(
            f"{task.name}: MLPilot trains binary tasks only; this one is {task.task_type.value}"
        )
    days = task.timedelta.days
    if task.timedelta != pd.Timedelta(days=days) or days < 1:
        raise BenchError(f"{task.name}: the horizon {task.timedelta} is not a whole number of days")
    db = task.get_db()  # rows up to the test timestamp, the task's hidden columns removed
    graph, tables = _database(db)
    entity = db.table_dict[task.entity_table]
    if entity.pkey_col is None:
        raise BenchError(f"{task.name}: the entity table {task.entity_table} has no primary key")
    parts = {s: _label_rows(task, s) for s in ("train", "val", "test")}
    labels = pd.concat(parts.values(), ignore_index=True)
    labels["label_window_end"] = labels[CUTOFF] + task.timedelta
    val_ts, test_ts = pd.Timestamp(dataset.val_timestamp), pd.Timestamp(dataset.test_timestamp)
    spec = from_dict(
        {
            "name": task.name.lower().replace("-", "_"),
            "description": f"RelBench task {task.name}; the labels are RelBench's own",
            "entity": {"table": task.entity_table, "key": entity.pkey_col},
            "target": {"type": "binary"},
            "horizon": f"{days}d",
            "cutoffs": {
                "start": labels[CUTOFF].min().date(),
                "end": labels[CUTOFF].max().date(),
                "every": f"{days}d",
            },
            "split": {"val_from": val_ts.date(), "test_from": test_ts.date()},
            "metric": "roc_auc",
        }
    )
    plan = TemporalSplitPlan(val_from=val_ts.to_pydatetime(), test_from=test_ts.to_pydatetime())
    name = getattr(getattr(dataset, "manifest", None), "name", None) or str(dataset)
    return Bench(
        name, task, spec, graph, tables, labels, plan, {s: len(p) for s, p in parts.items()}
    )


def _label_rows(task: Any, split: str) -> pd.DataFrame:
    table = task.get_table(split)  # the test table comes without its target
    df = table.df
    label = df[task.target_col].astype(int).to_numpy() if task.target_col in df else 0
    return pd.DataFrame(
        {
            ENTITY: df[task.entity_col].astype("int64").to_numpy(),
            CUTOFF: pd.to_datetime(df[task.time_col]).astype("datetime64[ns]").to_numpy(),
            "label": label,
        }
    )


def _sql_type(t: pa.DataType) -> str:
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_integer(t):
        return "BIGINT"
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "DOUBLE"
    if pa.types.is_timestamp(t):
        return "TIMESTAMP"
    if pa.types.is_date(t):
        return "DATE"
    return "VARCHAR"


def _database(db: Any) -> tuple[SchemaGraph, dict[str, pa.Table]]:
    """RelBench's tables, keys and time columns as an MLPilot schema graph and Arrow tables.

    The keys and time columns are RelBench's (declared, not guessed). A table without a time
    column is a static table: RelBench lets every cutoff read it, so MLPilot does too.
    """
    tables: dict[str, pa.Table] = {}
    graph_tables: list[Table] = []
    edges: list[Edge] = []
    for name, t in db.table_dict.items():
        df = t.df.copy()
        for col in t.fkey_col_to_pkey_table:
            df[col] = df[col].astype("Int64")  # a dangling key is missing, not a float
        data = pa.Table.from_pandas(df, preserve_index=False)
        tables[name] = data
        keys = {*t.fkey_col_to_pkey_table, *([t.pkey_col] if t.pkey_col else [])}
        columns = [
            Column(
                name=f.name,
                type=_sql_type(f.type),
                nullable=True,
                hint=column_hint(f.name, type_kind(_sql_type(f.type)), is_key=f.name in keys),
                is_primary_key=f.name == t.pkey_col,
            )
            for f in data.schema
        ]
        graph_tables.append(
            Table(
                key=name,
                name=name,
                db_schema="main",
                row_count=len(df),
                primary_key=[t.pkey_col] if t.pkey_col else [],
                columns=columns,
                time_column=t.time_col,
                time_column_source="user" if t.time_col else None,
                is_static=True if t.time_col is None else None,
            )
        )
        for fkey, parent in t.fkey_col_to_pkey_table.items():
            parent_key = db.table_dict[parent].pkey_col
            if parent_key is None:
                raise BenchError(f"{name}.{fkey} points to {parent}, which has no primary key")
            edges.append(
                Edge(
                    from_table=name,
                    from_columns=[fkey],
                    to_table=parent,
                    to_columns=[parent_key],
                    source="declared",
                    confidence=1.0,
                    cardinality="1:N",
                )
            )
    return SchemaGraph(dialect="duckdb", tables=graph_tables, edges=edges), tables


# -- the language model ----------------------------------------------------------------------


def llm_ready(model: str | None) -> bool:
    """True if ``provider:model`` names a provider whose key (or URL) is in the environment."""
    if not model:
        return False
    provider, _ = parse_entry(model)
    if provider not in PROVIDER_FACTORIES:  # the stub
        return False
    return bool(os.environ.get(PROVIDER_FACTORIES[provider][0]))


def make_gateway(model: str | None) -> AIGateway:
    """The gateway of an LLM run: keys from the environment, the SQL tier pinned to ``model``.

    No key for it means the offline stub answers, which the result records (``llm.answered``).
    """
    keys = SimpleNamespace(
        **{s: os.environ.get(s) or None for s, _, _ in PROVIDER_FACTORIES.values()}
    )
    router = TaskRouter(validate_tiers({"sql": [model]})) if model else TaskRouter()
    return AIGateway(config=keys, router=router)


class Recorded:
    """A gateway that keeps the record of every structured call it answers (for cost and tokens)."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.calls: list[dict[str, Any]] = []

    async def complete_structured_result(self, *args: Any, **kwargs: Any) -> Any:
        result = await self.gateway.complete_structured_result(*args, **kwargs)
        self.calls.append(result.meta())
        return result

    def summary(self, requested: str | None) -> dict[str, Any]:
        modes = Counter(str(c["decision_mode"]) for c in self.calls)
        return {
            "requested_model": requested,
            "answered": modes.get("llm", 0) > 0,
            "calls": len(self.calls),
            "answered_by": sorted({f"{c['provider']}:{c['model']}" for c in self.calls}),
            "decision_modes": dict(modes),
            # the gateway counts words, not tokens, and prices them from its model table
            "tokens_in_estimated": sum(int(c["tokens_in"]) for c in self.calls),
            "tokens_out_estimated": sum(int(c["tokens_out"]) for c in self.calls),
            "cost_usd_estimated": round(sum(float(c["cost_usd"]) for c in self.calls), 6),
        }


# -- running a method ------------------------------------------------------------------------


def _keys(labels: pd.DataFrame, rows: np.ndarray) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(labels.iloc[rows][[ENTITY, CUTOFF]])


def _aligned(scores: pd.Series, task: Any, table: Any) -> np.ndarray:
    """Scores in the order of RelBench's label table, which is the order ``evaluate`` expects."""
    want = pd.MultiIndex.from_arrays(
        [
            table.df[task.entity_col].astype("int64"),
            pd.to_datetime(table.df[task.time_col]).astype("datetime64[ns]"),
        ]
    )
    out = scores.reindex(want)
    if out.isna().any():
        raise BenchError(f"{int(out.isna().sum())} of {len(out)} RelBench rows have no score")
    return out.to_numpy(dtype="float64")


async def _agent(
    bench: Bench, baseline: BaselineResult, settings: Settings, gateway: Any
) -> loop.RunOutcome:
    flat, _ = flatten_graph(bench.graph, bench.spec.entity.table)
    assert baseline.labels is not None
    with FeatureEngine(
        flat,
        typed_times(bench.tables, flat),
        baseline.labels,
        data_version_id=bench.dataset,
        label_version=bench.spec.name,
        seed=settings.seed,
    ) as engine:
        proposer = FeatureProposer(
            spec=bench.spec,
            graph=bench.graph,
            tables=bench.tables,
            baseline=baseline,
            gateway=gateway,
            builder=ContextBuilder(),
            engine=engine,
        )
        config = loop.RunConfig(
            max_rounds=settings.max_rounds,
            patience=settings.patience,
            budget=Budget(settings.max_cost_usd, settings.max_seconds),
            seed=settings.seed,
        )
        # the test rows are not scored here: the harness scores them once, with RelBench's metric
        return await loop.run_loop(baseline, proposer, config, score_test=False)


async def run_method(
    bench: Bench, method: str, settings: Settings, gateway: Any = None
) -> dict[str, Any]:
    """One run of one method; returns the result record (see ``write_result``)."""
    method = ALIASES.get(method, method)
    if method not in METHODS:
        raise BenchError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    started = time.monotonic()
    baseline = build_baseline(
        bench.spec,
        bench.graph,
        bench.tables,
        bench.labels,
        bench.plan,
        seed=settings.seed,
        groups=frozenset({"attribute"}) if method == "a" else None,
    )
    assert baseline.frame is not None and baseline.labels is not None
    assert baseline.temporal is not None and baseline.model is not None
    split = baseline.temporal
    frame, model, best = baseline.frame, baseline.model, int(baseline.metrics["best_iteration"])
    llm: dict[str, Any] | None = None
    agent: dict[str, Any] | None = None
    if method in ("c", "d"):
        recorded = Recorded(gateway or make_gateway(settings.model))
        outcome = await _agent(bench, baseline, settings, recorded)
        frame, model = outcome.champion.frame, outcome.champion.model
        best = outcome.champion.best_iteration
        llm = recorded.summary(settings.model)
        agent = {
            "stop_reason": outcome.stop_reason,
            "rounds": [_round(r) for r in outcome.rounds],
            "accepted": [r.record.proposal.name for r in outcome.accepted if r.record.proposal],
            "budget_used": outcome.used,
        }
    fitted = time.monotonic()
    y = baseline.labels["label"].astype(int).to_numpy()  # test rows hold 0: never read
    task = bench.task
    val_scores = pd.Series(model.predict_proba(frame.iloc[split.val])[:, 1])
    val_scores.index = _keys(baseline.labels, split.val)
    final = loop.refit_on_train_val(frame, best, y, split, settings.seed)
    test_scores = pd.Series(np.asarray(final.predict_proba(frame.iloc[split.test]))[:, 1])
    test_scores.index = _keys(baseline.labels, split.test)
    if len(split.test) != bench.rows["test"] or len(split.val) != bench.rows["val"]:
        raise BenchError(
            f"the split lost rows: val {len(split.val)} of {bench.rows['val']}, "
            f"test {len(split.test)} of {bench.rows['test']}"
        )
    val_table = task.get_table("val", mask_input_cols=False)
    val = task.evaluate(_aligned(val_scores, task, val_table), val_table)
    # the one place the test labels are read: RelBench's evaluator, once, on the final predictions
    test = task.evaluate(_aligned(test_scores, task, task.get_table("test")))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "dataset": bench.dataset,
        "task": task.name,
        "method": method,
        "method_name": METHODS[method],
        "seed": settings.seed,
        "val": _floats(val),
        "test": _floats(test),
        "rows": {**bench.rows, "split": baseline.split},
        "features": {
            "n": len(frame.columns),
            "baseline_candidates": baseline.candidates,
            "baseline_kept": len(baseline.features),
            "baseline_dropped": baseline.dropped_counts(),
            "names": [str(c) for c in frame.columns][:50],
        },
        "budget": settings.budget() if method in ("c", "d") else None,
        "llm": llm,
        "agent": agent,
        "wall_seconds": {
            "total": round(time.monotonic() - started, 2),
            "fit": round(fitted - started, 2),
        },
        "environment": environment(),
    }


def _round(r: loop.Round) -> dict[str, Any]:
    p = r.record.proposal
    return {
        "round": r.number,
        "name": p.name if p else None,
        "status": r.record.status,
        "stage": r.record.stage,
        "attempts": [
            {"stage": a.stage, "reasons": [x[:200] for x in a.reasons[:3]]}
            for a in r.record.attempts
        ],
        "accepted": r.accepted,
        "mean_gain_pr_auc": r.gain.mean_gain if r.gain else None,
    }


def _floats(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def not_run(
    bench: Bench, method: str, settings: Settings, status: str, **fields: Any
) -> dict[str, Any]:
    """The record of a run that was skipped (``reason``) or failed (``error``); it has no numbers."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        **fields,
        "dataset": bench.dataset,
        "task": bench.task.name,
        "method": method,
        "method_name": METHODS[method],
        "seed": settings.seed,
        "environment": environment(),
    }


def environment() -> dict[str, Any]:
    import relbench

    return {
        "relbench": relbench.__version__,
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
        "mlpilot_commit": _commit(),
        "ci_run": os.environ.get("GITHUB_RUN_ID"),
    }


def _commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def result_path(out_dir: Path, record: dict[str, Any]) -> Path:
    return (
        out_dir
        / record["dataset"]
        / record["task"]
        / f"{record['method']}_seed{record['seed']}.json"
    )


def write_result(out_dir: Path, record: dict[str, Any]) -> Path:
    path = result_path(out_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path
