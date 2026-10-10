"""The run manifest: everything needed to reproduce one experiment, and its replay.

Written once when an experiment finishes (app/services/experiment_service.py) and
stored on it. `replay` rebuilds the model from the manifest alone: same data version,
same features in the same order, same engine parameters, same split and seed, and no
LLM call. AGENTS.md section 6 lists what a run must record; this is where it lives.

The manifest is built from named fields only, never by copying a whole parameters dict,
so nothing that happens to sit on a run (an error text, a connection detail) can leak
into it. LLM calls are recorded as provider, model, tokens, cost and decision mode.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ml.data.snapshot import DataDescription
from ml.experiments.schema import ExperimentResult, ExperimentSpec

MANIFEST_VERSION = 1
REPO_DIR = Path(__file__).resolve().parents[3]
# Packages whose versions can change a model or a metric.
PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "lightgbm",
    "xgboost",
    "optuna",
    "sqlglot",
    "pydantic",
)


class FeatureEntry(BaseModel):
    """One engineered feature, in the order it was added."""

    name: str
    kind: str = "formula"
    formula: str


class EngineEntry(BaseModel):
    name: str
    version: str
    params: dict[str, Any] = Field(description="The parameters the final model was trained with")
    tuning: str | None = None


class LlmEntry(BaseModel):
    """One LLM call that shaped the run (the hypothesis). No prompt or response text."""

    role: str
    provider: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    decision_mode: str | None = None
    providers_failed: int = 0


class RunManifest(BaseModel):
    manifest_version: int = MANIFEST_VERSION
    experiment_id: str
    parent_id: str | None
    created_at: datetime

    # Software
    mlpilot_version: str
    git_sha: str | None = Field(description="None when the code is not a git checkout")
    python_version: str
    packages: dict[str, str | None]

    # Data and task
    data_version_id: str | None
    data: DataDescription | None = Field(
        None,
        description="For a database: snapshot or live, as_of, per-table row counts and latest event times",
    )
    task: dict[str, Any] = Field(description="Target column, classes and positive class")
    split_plan: dict[str, Any]
    seeds: dict[str, int]

    # Model and features
    engine: EngineEntry
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    features: list[FeatureEntry]

    # Decisions
    acceptance_rule: dict[str, Any] | None = None
    acceptance: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    threshold: dict[str, Any] | None = None

    # Results
    metrics: dict[str, dict[str, float | None]] = Field(description="{'val': {...}, 'test': {...}}")
    llm: list[LlmEntry] = Field(default_factory=list)
    timings: dict[str, float]


@cache
def _software() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _git_sha() -> str | None:
    if not (REPO_DIR / "backend" / "ml").is_dir():
        return None  # installed from a wheel: whatever repository the venv sits in is not ours
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def split_metrics(metrics: dict[str, float | None]) -> dict[str, dict[str, float | None]]:
    """{'val_f1': .., 'test_f1': .., 'f1': ..} -> {'val': {'f1': ..}, 'test': {'f1': ..}}."""
    out: dict[str, dict[str, float | None]] = {"val": {}, "test": {}}
    for key, value in metrics.items():
        part, _, name = key.partition("_")
        if part in out and name:
            out[part][name] = value
    return out


def build_manifest(
    result: ExperimentResult,
    *,
    data_version_id: str | None,
    preprocessing_config: dict[str, Any] | None,
    mlpilot_version: str,
    data: DataDescription | None = None,
) -> RunManifest:
    """The manifest of a completed experiment, from named fields of its result."""
    p = result.parameters
    engine = p.get("engine") or {"name": result.model_name, "version": "unknown"}
    features = [FeatureEntry(name=f["name"], formula=f["formula"]) for f in p.get("features") or []]
    if p.get("feature_name") and p.get("formula"):
        features.append(FeatureEntry(name=p["feature_name"], formula=p["formula"]))
    plan = p.get("split_plan") or {}
    llm = []
    if p.get("hypothesis_llm"):
        h = p["hypothesis_llm"]
        llm.append(
            LlmEntry(
                role="hypothesis",
                provider=h.get("provider"),
                model=h.get("model"),
                tokens_in=h.get("tokens_in"),
                tokens_out=h.get("tokens_out"),
                cost_usd=h.get("cost_usd"),
                decision_mode=h.get("decision_mode"),
                providers_failed=len(h.get("errors") or []),
            )
        )
    return RunManifest(
        experiment_id=result.id,
        parent_id=result.parent_id,
        created_at=datetime.now(UTC),
        mlpilot_version=mlpilot_version,
        **_software(),
        data_version_id=data_version_id,
        data=data,
        task={
            "target_column": p.get("target_column"),
            **(p.get("target_encoding") or {}),
        },
        split_plan=plan,
        seeds={"split": int(plan.get("seed", 42)), "model": int(plan.get("seed", 42))},
        engine=EngineEntry(
            name=engine["name"],
            version=engine["version"],
            params={**(p.get("model_params") or {}), **(p.get("best_params") or {})},
            tuning=p.get("tuning"),
        ),
        preprocessing_config=preprocessing_config or {},
        features=features,
        acceptance_rule=p.get("acceptance_rule"),
        acceptance=p.get("acceptance"),
        calibration=p.get("calibration"),
        threshold=p.get("threshold"),
        metrics=split_metrics(result.metrics),
        llm=llm,
        timings={"runtime_seconds": result.runtime_seconds},
    )


def replay_spec(manifest: RunManifest, dataset_path: str) -> ExperimentSpec:
    """An experiment spec that retrains exactly what the manifest describes.

    All features are applied as accepted ones (no candidate, so no keep/reject test),
    the engine parameters are used as recorded and not tuned again.
    """
    task = manifest.task
    return ExperimentSpec(
        id=f"replay-{manifest.experiment_id}",
        parent_id=manifest.experiment_id,
        project_id="replay",
        dataset_version=dataset_path,
        hypothesis="Replay from the run manifest",
        change_description=f"Replay of {manifest.experiment_id}",
        model_name=manifest.engine.name,
        parameters={
            "target_column": task["target_column"],
            "positive_class": task.get("positive_class"),
            "features": [f.model_dump(include={"name", "formula"}) for f in manifest.features],
            "model_params": dict(manifest.engine.params),
            "replay": True,
        },
        validation_config=dict(manifest.split_plan),
        preprocessing_config=dict(manifest.preprocessing_config),
        feature_set=[f.name for f in manifest.features],
    )


async def replay(
    manifest: RunManifest, dataset_path: str, load: Callable[[str], pd.DataFrame]
) -> ExperimentResult:
    """Retrain and re-score the run from its manifest alone (no LLM, no tuning)."""
    from ml.experiments.executor import LocalExperimentExecutor

    return await LocalExperimentExecutor(data_loader_func=load).run(
        replay_spec(manifest, dataset_path)
    )
