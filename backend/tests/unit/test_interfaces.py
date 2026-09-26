"""Tests that all ABCs raise NotImplementedError when not implemented."""
from __future__ import annotations

import pytest

from ml.core.interfaces import (
    DatasetProvider,
    DataPreparationProvider,
    AIProvider,
    ExperimentRunner,
    ModelProvider,
    MetricProvider,
    ValidationStrategy,
    ReportProvider,
    DeploymentProvider,
)


# ── Concrete subclasses that implement nothing ────────────────────────────────

class ConcreteDatasetProvider(DatasetProvider):
    def load(self, path, **kw): raise NotImplementedError
    def supports(self, path): raise NotImplementedError
    def schema(self, df): raise NotImplementedError


class ConcreteDataPreparationProvider(DataPreparationProvider):
    def profile(self, df): raise NotImplementedError
    def assess_readiness(self, df, target): raise NotImplementedError
    def detect_leakage(self, df, target): raise NotImplementedError
    def prepare(self, df, config): raise NotImplementedError
    def export_pipeline(self, pipeline, path): raise NotImplementedError


class ConcreteAIProvider(AIProvider):
    async def complete(self, prompt, system="", model=None, max_tokens=1024, temperature=0.7): raise NotImplementedError
    async def complete_structured(self, prompt, schema, system="", model=None, max_tokens=2048): raise NotImplementedError
    def estimate_cost(self, pt, ct, model=None): raise NotImplementedError
    async def list_models(self): raise NotImplementedError
    async def health_check(self): raise NotImplementedError


class ConcreteExperimentRunner(ExperimentRunner):
    async def run(self, spec): raise NotImplementedError
    def validate_spec(self, spec): raise NotImplementedError
    async def get_status(self, eid): raise NotImplementedError
    async def cancel(self, eid): raise NotImplementedError


class ConcreteModelProvider(ModelProvider):
    def train(self, X, y, params): raise NotImplementedError
    def predict(self, model, X): raise NotImplementedError
    def save(self, model, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError
    def get_feature_importance(self, model, names): raise NotImplementedError


class ConcreteMetricProvider(MetricProvider):
    def compute_classification(self, yt, yp, yprob=None): raise NotImplementedError
    def compute_regression(self, yt, yp): raise NotImplementedError
    def compare(self, baseline, challenger, metric): raise NotImplementedError


class ConcreteValidationStrategy(ValidationStrategy):
    def split(self, df, target): raise NotImplementedError
    def describe(self): raise NotImplementedError
    def get_config(self): raise NotImplementedError


class ConcreteReportProvider(ReportProvider):
    async def generate(self, project_id, exp_ids, format="markdown", **kw): raise NotImplementedError


class ConcreteDeploymentProvider(DeploymentProvider):
    async def deploy(self, model_path, project_id, config): raise NotImplementedError
    async def get_endpoint(self, dep_id): raise NotImplementedError
    async def get_status(self, dep_id): raise NotImplementedError
    async def teardown(self, dep_id): raise NotImplementedError


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dataset_provider_is_abstract():
    import inspect
    assert inspect.isabstract(DatasetProvider)

def test_data_preparation_provider_is_abstract():
    import inspect
    assert inspect.isabstract(DataPreparationProvider)

def test_ai_provider_is_abstract():
    import inspect
    assert inspect.isabstract(AIProvider)

def test_experiment_runner_is_abstract():
    import inspect
    assert inspect.isabstract(ExperimentRunner)

def test_model_provider_is_abstract():
    import inspect
    assert inspect.isabstract(ModelProvider)

def test_metric_provider_is_abstract():
    import inspect
    assert inspect.isabstract(MetricProvider)

def test_validation_strategy_is_abstract():
    import inspect
    assert inspect.isabstract(ValidationStrategy)

def test_report_provider_is_abstract():
    import inspect
    assert inspect.isabstract(ReportProvider)

def test_deployment_provider_is_abstract():
    import inspect
    assert inspect.isabstract(DeploymentProvider)


def test_dataset_provider_concrete_raises():
    p = ConcreteDatasetProvider()
    with pytest.raises(NotImplementedError):
        p.load("x")

def test_data_prep_profile_raises():
    p = ConcreteDataPreparationProvider()
    with pytest.raises(NotImplementedError):
        p.profile(None)

def test_data_prep_assess_raises():
    p = ConcreteDataPreparationProvider()
    with pytest.raises(NotImplementedError):
        p.assess_readiness(None, "target")

def test_data_prep_detect_leakage_raises():
    p = ConcreteDataPreparationProvider()
    with pytest.raises(NotImplementedError):
        p.detect_leakage(None, "target")

def test_model_provider_train_raises():
    p = ConcreteModelProvider()
    with pytest.raises(NotImplementedError):
        p.train(None, None, {})

def test_metric_provider_compute_classification_raises():
    p = ConcreteMetricProvider()
    with pytest.raises(NotImplementedError):
        p.compute_classification([], [])

def test_validation_strategy_split_raises():
    p = ConcreteValidationStrategy()
    with pytest.raises(NotImplementedError):
        p.split(None, "target")


@pytest.mark.asyncio
async def test_ai_provider_complete_raises():
    p = ConcreteAIProvider()
    with pytest.raises(NotImplementedError):
        await p.complete("hi")


@pytest.mark.asyncio
async def test_experiment_runner_run_raises():
    p = ConcreteExperimentRunner()
    with pytest.raises(NotImplementedError):
        await p.run(None)


@pytest.mark.asyncio
async def test_report_provider_generate_raises():
    p = ConcreteReportProvider()
    with pytest.raises(NotImplementedError):
        await p.generate("p1", [])


@pytest.mark.asyncio
async def test_deployment_provider_deploy_raises():
    p = ConcreteDeploymentProvider()
    with pytest.raises(NotImplementedError):
        await p.deploy("path", "p1", {})
