"""Single-table sanity benchmark on OpenML: LightGBM, MLPilot and AutoGluon (issue #67).

For a CSV user the question is where MLPilot stands against the strongest free tool. Ten OpenML
binary-classification datasets (``DATASETS``, pinned by OpenML dataset id) are run by three methods
on **one** stratified train / validation / test split per dataset and seed (60 / 20 / 20, made by
the product's own ``make_splits``), and the test rows are scored **once** per run, by ``score``:

    lightgbm  the MLPilot baseline of a table (``build_table_baseline``): its preparation, LightGBM
              stopped early on the validation rows, refit on train + validation
    mlpilot   the same baseline, then the run loop (``ml.agents.run_loop``) with formula features
              proposed by a model and kept only when the paired rule accepts them. Without a model
              and its key the offline stub answers, proposes nothing (``llm.answered`` is false)
              and this is the lightgbm method run through the loop: the record says so
    autogluon ``TabularPredictor`` with a time limit, fitted on train + validation (it makes its
              own internal validation split), on the raw columns

Nothing here chooses anything on the test rows: they are scored after every decision is made, and
the AutoGluon fit never sees them. A result file is written from a run; ``openml_report`` reads only
result files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ai.context_builder import ContextBuilder
from ml.agents import run_loop as loop
from ml.data.profiling.profiler import DataProfiler
from ml.experiments.planner import ExperimentPlanner
from ml.features.baseline import BaselineResult
from ml.features.engine import Budget
from ml.features.formula_proposer import FormulaProposer
from ml.features.table_baseline import TableTask, build_table_baseline
from ml.validation.splits import SplitPlan, make_splits

# The gateway helpers of the RelBench harness (benchmarks/relbench is on the path, see openml_run).
from relbench_harness import Recorded, make_gateway  # isort: skip

SCHEMA_VERSION = 1
METHODS = {
    "lightgbm": "LightGBM (the MLPilot baseline)",
    "mlpilot": "MLPilot (baseline + the run loop)",
    "autogluon": "AutoGluon (time-limited)",
}
ENV_MODEL = "MLPILOT_BENCH_CHEAP_MODEL"
REPO = Path(__file__).resolve().parents[2]
# OpenML dataset ids (all binary, 500 to 5500 rows). The licence is read from OpenML when a dataset
# is loaded and written into every record, not asserted here.
DATASETS: dict[int, str] = {
    37: "diabetes",
    31: "credit-g",
    29: "credit-approval",
    1464: "blood-transfusion-service-center",
    1494: "qsar-biodeg",
    1480: "ilpd",
    1489: "phoneme",
    44: "spambase",
    1067: "kc1",
    1487: "ozone-level-8hr",
}


class BenchError(ValueError):
    """The run cannot be done as asked (never a silent partial result)."""


@dataclass(frozen=True)
class Settings:
    """What one run may spend. The language-model budget is the same as RelBench's."""

    seed: int = 0
    max_rounds: int = 10
    patience: int = 5
    max_cost_usd: float | None = 1.0
    max_seconds: float | None = 600.0
    model: str | None = None  # "provider:model" for the mlpilot method
    ag_seconds: int = 90  # AutoGluon time limit for the fit, per run
    ag_preset: str = "high_quality"


@dataclass
class Data:
    """One dataset as the methods see it: a frame with a 0/1 target column."""

    meta: dict[str, Any]
    frame: pd.DataFrame
    target: str


# -- data ----------------------------------------------------------------------------------------


def binary_target(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The frame with its target as 0/1, 1 being the rarer class (ties: the later label)."""
    y = frame[target]
    if y.isna().any():
        raise BenchError(f"the target {target!r} has missing values")
    counts = y.astype(str).value_counts()
    if len(counts) != 2:
        raise BenchError(
            f"the target {target!r} has {len(counts)} classes; this benchmark is binary"
        )
    positive = min(sorted(counts.index, reverse=True), key=lambda c: counts[c])
    out = frame.copy()
    out[target] = (y.astype(str) == positive).astype(int)
    return out, {"positive_class": positive, "base_rate": round(float(out[target].mean()), 4)}


def load_openml(dataset_id: int) -> Data:
    """Download one OpenML dataset (cached by openml-python) and read its licence from OpenML."""
    import openml

    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    target = ds.default_target_attribute
    if not target or "," in target:
        raise BenchError(f"OpenML {dataset_id} has no single default target ({target!r})")
    X, y, _, _ = ds.get_data(target=target)
    frame, info = binary_target(
        pd.concat([pd.DataFrame(X), pd.Series(y, name=target)], axis=1), target
    )
    meta = {
        "source": "openml",
        "id": dataset_id,
        "name": ds.name,
        "version": ds.version,
        "licence": getattr(ds, "licence", None),
        "url": f"https://www.openml.org/d/{dataset_id}",
        "target": target,
        "n_rows": len(frame),
        "n_features": frame.shape[1] - 1,
        **info,
    }
    return Data(meta, frame, target)


def load_csv(path: Path, target: str) -> Data:
    frame, info = binary_target(pd.read_csv(path), target)
    meta = {
        "source": "csv",
        "id": path.stem,
        "name": path.stem,
        "licence": None,
        "target": target,
        "n_rows": len(frame),
        "n_features": frame.shape[1] - 1,
        **info,
    }
    return Data(meta, frame, target)


# -- the one split and the one scoring ------------------------------------------------------------


def split_rows(frame: pd.DataFrame, target: str, seed: int) -> tuple[np.ndarray, ...]:
    """Train, validation and test row positions: stratified, 60 / 20 / 20, fixed by the seed."""
    s = make_splits(frame[target], SplitPlan(strategy="holdout", seed=seed))
    return s.train, s.val, s.test


def split_hash(*parts: np.ndarray) -> str:
    """A fingerprint of the split, so the report can check that every method had the same rows."""
    h = hashlib.sha1()
    for p in parts:
        h.update(np.asarray(p, dtype="int64").tobytes())
    return h.hexdigest()[:12]


def score(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """The test metrics. The one place the test rows are scored; every run calls it once."""
    return {
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "base_rate": float(np.mean(y)),
        "n": float(len(y)),
    }


# -- running a method -----------------------------------------------------------------------------


def ag_scores(
    frame: pd.DataFrame, target: str, fit: np.ndarray, test: np.ndarray, settings: Settings
) -> tuple[np.ndarray, dict[str, Any]]:
    """AutoGluon fitted on ``fit`` rows (train + validation), probabilities for the test rows."""
    from autogluon.tabular import TabularPredictor

    with tempfile.TemporaryDirectory(prefix="ag-") as tmp:
        predictor = TabularPredictor(
            label=target, problem_type="binary", eval_metric="roc_auc", path=tmp, verbosity=0
        ).fit(
            frame.iloc[fit],
            time_limit=settings.ag_seconds,
            presets=settings.ag_preset,
        )
        prob = predictor.predict_proba(frame.iloc[test].drop(columns=[target]))
        best = predictor.model_best
        n_models = len(predictor.model_names())
    return prob[1].to_numpy(dtype="float64"), {
        "name": "AutoGluon",
        "version": importlib.metadata.version("autogluon.tabular"),
        "time_limit_seconds": settings.ag_seconds,
        "preset": settings.ag_preset,
        "eval_metric": "roc_auc",
        "best_model": best,
        "models_fitted": n_models,
        "seed": "AutoGluon's own default; the seed changes the split only",
    }


async def run_mlpilot_loop(
    data: Data, baseline: BaselineResult, task: TableTask, settings: Settings, gateway: Any
) -> loop.RunOutcome:
    """The run loop of a table, as ``table_run_service`` runs it, with one difference: the profile
    the model hears about is built from the train and validation rows, so even the test rows' class
    balance stays out of the prompt. The test rows are not scored here."""
    pool = data.frame.iloc[task.split.pool]
    proposer = FormulaProposer(
        planner=ExperimentPlanner(gateway, ContextBuilder()),
        profile=DataProfiler().profile(pool, target_column=data.target),
        task=task,
        baseline=baseline,
    )
    config = loop.RunConfig(
        max_rounds=settings.max_rounds,
        patience=settings.patience,
        budget=Budget(settings.max_cost_usd, settings.max_seconds),
        seed=settings.seed,
    )
    return await loop.run_loop(baseline, proposer, config, score_test=False)


async def run_method(
    data: Data, method: str, settings: Settings, gateway: Any = None
) -> dict[str, Any]:
    """One run of one method on one dataset; returns the result record (see ``write_result``)."""
    if method not in METHODS:
        raise BenchError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    started = time.monotonic()
    frame, target = data.frame, data.target
    train, val, test = split_rows(frame, target, settings.seed)
    y = frame[target].to_numpy(dtype=int)
    engine: dict[str, Any]
    llm: dict[str, Any] | None = None
    agent: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    if method == "autogluon":
        # fitted on train + validation, as the final models of the other two are; never on test
        fit_rows = np.sort(np.concatenate([train, val]))
        prob, engine = ag_scores(frame, target, fit_rows, test, settings)
    else:
        baseline, task = build_table_baseline(frame, target, seed=settings.seed, positive_class=1)
        assert baseline.temporal is not None and baseline.frame is not None
        if not (
            np.array_equal(task.split.train, train)
            and np.array_equal(task.split.val, val)
            and np.array_equal(task.split.test, test)
        ):
            raise BenchError("the baseline's split differs from the benchmark's split")
        champion_frame, best = baseline.frame, int(baseline.metrics["best_iteration"])
        if method == "mlpilot":
            recorded = Recorded(gateway or make_gateway(settings.model))
            outcome = await run_mlpilot_loop(data, baseline, task, settings, recorded)
            champion_frame, best = outcome.champion.frame, outcome.champion.best_iteration
            llm = recorded.summary(settings.model)
            agent = {
                "stop_reason": outcome.stop_reason,
                "rounds": len(outcome.rounds),
                "accepted": [r.record.proposal.name for r in outcome.accepted if r.record.proposal],
                "budget": {
                    "max_rounds": settings.max_rounds,
                    "patience": settings.patience,
                    "max_cost_usd": settings.max_cost_usd,
                    "max_seconds": settings.max_seconds,
                },
                "budget_used": outcome.used,
            }
        model = loop.refit_on_train_val(champion_frame, best, y, task.split, settings.seed)
        prob = np.asarray(model.predict_proba(champion_frame.iloc[test]))[:, 1]
        engine = {"name": "LightGBM", "version": lgb.__version__, "n_estimators": max(10, best)}
        features = {
            "n": champion_frame.shape[1],
            "excluded_by_preparation": task.report["excluded_features"],
        }
    # the one place the test labels are read, after every fit and every decision
    result = score(y[test], prob)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "dataset": data.meta,
        "method": method,
        "method_name": METHODS[method],
        "seed": settings.seed,
        "test": result,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "split_hash": split_hash(train, val, test),
        "engine": engine,
        "features": features,
        "llm": llm,
        "agent": agent,
        "wall_seconds": round(time.monotonic() - started, 2),
        "environment": environment(),
    }


def not_run(
    data: Data, method: str, settings: Settings, status: str, **fields: Any
) -> dict[str, Any]:
    """The record of a run that failed (``error``): it has no numbers."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        **fields,
        "dataset": data.meta,
        "method": method,
        "method_name": METHODS[method],
        "seed": settings.seed,
        "environment": environment(),
    }


# -- records ------------------------------------------------------------------------------------


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    return {
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "openml": package_version("openml"),
        "autogluon": package_version("autogluon.tabular"),
        "torch": package_version("torch"),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
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


def write_result(out_dir: Path, record: dict[str, Any]) -> Path:
    name = f"{record['dataset']['id']}-{record['dataset']['name']}"
    path = out_dir / name / f"{record['method']}_seed{record['seed']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path
