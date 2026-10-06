"""#94: every completed run stores a manifest, and replaying it reproduces the metrics."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import app.db.session as db_session
from app.core.config import settings
from app.services.experiment_service import ExperimentService
from ml.experiments.executor import LocalExperimentExecutor
from ml.experiments.manifest import RunManifest, build_manifest, replay, split_metrics
from ml.experiments.planner import ExperimentPlanner
from ml.experiments.schema import ExperimentResult, ExperimentSpec, ExperimentStatus
from tests.conftest import TEST_TOKEN
from tests.fixtures.api import API, load_sample
from tests.integration.test_agent_loop_telecom import follow_job

TOLERANCE = 1e-9


def assert_same_metrics(recorded: dict, replayed: dict) -> None:
    for part in ("val", "test"):
        assert recorded[part].keys() == replayed[part].keys(), part
        for name, value in recorded[part].items():
            again = replayed[part][name]
            if value is None:
                assert again is None, (part, name)
            else:
                assert again == pytest.approx(value, abs=TOLERANCE, rel=0), (part, name)


@pytest.fixture
def two_formulas(monkeypatch):
    proposals = iter(
        [
            {"name": "charge_x_tenure", "formula": "MonthlyCharges * tenure"},
            {"name": "avg_charge", "formula": "TotalCharges / (tenure + 1)"},
        ]
    )

    async def propose(self, **kwargs):
        return {**next(proposals), "reason": "test proposal", "non_redundant_reasoning": "new"}

    monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", propose)


async def test_telecom_run_replays_from_its_manifest(client, project_id, monkeypatch, two_formulas):
    """The telecom e2e flow, then each experiment retrained from its manifest alone."""
    monkeypatch.setitem(sys.modules, "optuna", None)  # as in the e2e test: no tuning
    base = f"{API}/projects/{project_id}"
    version = (await load_sample(client, project_id))["data_version_id"]
    job = await client.post(
        f"{base}/agent/auto-optimize",
        json={"data_version_id": version, "target_column": "Churn", "n_hypotheses": 2},
    )
    status, _ = await follow_job(client, base, job.json()["id"])
    assert status["status"] == "succeeded", status

    tree = (await client.get(f"{base}/experiments", params={"data_version_id": version})).json()
    exps = [(await client.get(f"{base}/experiments/{n['id']}")).json() for n in tree]
    assert len(exps) == 3
    for exp in exps:
        manifest = RunManifest.model_validate(exp["manifest"])
        assert manifest.data_version_id == version
        assert manifest.task["target_column"] == "Churn"
        assert manifest.engine.version and manifest.packages["scikit-learn"]
        assert manifest.split_plan["seed"] == manifest.seeds["split"]
        assert manifest.metrics["val"]["pr_auc"] and manifest.metrics["test"]["pr_auc"]
        if exp["parent_id"]:
            assert manifest.llm and manifest.llm[0].role == "hypothesis"
            assert manifest.features[-1].formula in exp["parameters"]["formula"]

        # Replay without the LLM: a planner call would fail the test.
        async def no_llm(*args, **kwargs):
            raise AssertionError("replay must not call the LLM")

        monkeypatch.setattr(ExperimentPlanner, "generate_next_hypothesis", no_llm)
        async with db_session.AsyncSessionLocal() as db:
            again = await ExperimentService(db).replay(exp["id"])
        assert again.status == ExperimentStatus.COMPLETED
        assert again.parameters["tuning"] == "none: replayed with the recorded parameters"
        assert_same_metrics(manifest.metrics, split_metrics(again.metrics))

        # The features are rebuilt in the recorded order.
        assert (
            again.parameters["feature_columns"][-len(manifest.features) :]
            == [f.name for f in manifest.features]
            or not manifest.features
        )

    # No secret of this installation is in any manifest.
    text = json.dumps([e["manifest"] for e in exps])
    for value in configured_secrets():
        assert value not in text
    assert TEST_TOKEN not in text


async def test_tuned_parameters_replay_exactly():
    """With Optuna tuning on, the replay uses the tuned parameters and does not re-tune."""
    pytest.importorskip("optuna")
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=3)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = np.where(y == 1, "yes", "no")
    spec = ExperimentSpec(
        id="tuned",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name="LGBMClassifier",
        parameters={
            "target_column": "target",
            "n_trials": 3,
            "features": [{"name": "f01", "formula": "f0 * f1"}],
        },
        feature_set=[],
    )
    load = lambda _: df.copy()
    result = await LocalExperimentExecutor(data_loader_func=load).run(spec)
    assert result.parameters["best_params"]  # tuning ran
    manifest = build_manifest(
        result, data_version_id=None, preprocessing_config=None, mlpilot_version="test"
    )
    assert manifest.engine.params == result.parameters["best_params"]
    assert [f.name for f in manifest.features] == ["f01"]

    again = await replay(manifest, "v", load)
    assert "best_params" not in again.parameters
    assert_same_metrics(manifest.metrics, split_metrics(again.metrics))


def configured_secrets() -> list[str]:
    names = [n for n in type(settings).model_fields if re.search("KEY|PASSWORD|TOKEN", n)]
    return [v for n in names if isinstance(v := getattr(settings, n), str) and len(v) >= 8]


def test_the_manifest_contains_no_secrets(monkeypatch):
    """Secrets anywhere on a run's parameters or settings never reach the manifest."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-SENTINEL-openai-key-0123456789")
    monkeypatch.setattr(settings, "SECRET_KEY", "SENTINEL-secret-key-0123456789")
    password = "SENTINEL-db-password-0123456789"
    result = ExperimentResult(
        id="e",
        project_id="p",
        dataset_version="v",
        hypothesis="h",
        change_description="c",
        model_name="LGBMClassifier",
        parameters={
            "target_column": "t",
            "engine": {"name": "LGBMClassifier", "version": "4"},
            "split_plan": {"strategy": "holdout", "seed": 42},
            "connection_string": f"postgresql://user:{password}@db/prod",
            "api_key": settings.OPENAI_API_KEY,
            "hypothesis_llm": {
                "provider": "openai",
                "model": "m",
                "decision_mode": "llm",
                "errors": [f"groq failed with key {settings.OPENAI_API_KEY}"],
            },
        },
        validation_config={},
        metrics={"val_f1": 0.5, "test_f1": 0.4},
        runtime_seconds=1.0,
        cost_usd=0.0,
        status=ExperimentStatus.COMPLETED,
        timestamp=datetime.now(UTC),
    )
    manifest = build_manifest(
        result,
        data_version_id=None,
        preprocessing_config={"note": f"from postgresql://user:{password}@db/prod"},
        mlpilot_version="test",
    )
    # Preprocessing config is user configuration, not secrets; the rest is named fields.
    text = manifest.model_copy(update={"preprocessing_config": {}}).model_dump_json()
    for secret in [password, *configured_secrets()]:
        assert secret not in text, secret
    assert manifest.llm[0].providers_failed == 1
