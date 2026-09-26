"""All 9 provider ABCs for MLPilot."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Shared dataclasses
# ─────────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    """Result of dataset profiling."""
    rows: int
    columns: int
    missing_rate: float           # fraction of all cells that are missing
    duplicate_rows: int
    target_balance: dict[str, Any]  # class distribution for classification, stats for regression
    column_stats: dict[str, Any]    # per-column stats keyed by column name
    warnings: list[str] = field(default_factory=list)


@dataclass
class LeakageWarning:
    """A single data leakage warning."""
    column: str
    leakage_type: str   # 'target', 'temporal', 'entity', 'preprocessing'
    severity: str       # 'high', 'medium', 'low'
    reason: str
    suggested_action: str


@dataclass
class ReadinessReport:
    """Overall data readiness assessment."""
    data_quality_score: float       # 0.0 – 1.0
    leakage_risk: str               # 'high', 'medium', 'low', 'none'
    validation_risk: str            # 'high', 'medium', 'low', 'none'
    feature_risk: str               # 'high', 'medium', 'low', 'none'
    warnings: list[LeakageWarning] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Metadata for an AI language model."""
    name: str
    provider: str
    context_window: int
    cost_per_1k_prompt_tokens: float
    cost_per_1k_completion_tokens: float
    capabilities: list[str] = field(default_factory=list)
    is_free: bool = False


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None
    confusion_matrix: Optional[list[list[int]]] = None


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    mape: Optional[float] = None


@dataclass
class ValidationSplit:
    """A single train/validation split."""
    train_indices: list[int]
    val_indices: list[int]
    fold: int = 0


# ─────────────────────────────────────────────────────────────────
# 1. DatasetProvider
# ─────────────────────────────────────────────────────────────────

class DatasetProvider(ABC):
    """Abstract interface for loading datasets."""

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from *path* and return a DataFrame."""
        ...

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Return True if this provider can load the file at *path*."""
        ...

    @abstractmethod
    def schema(self, df: pd.DataFrame) -> dict[str, str]:
        """Return column → dtype mapping for the given DataFrame."""
        ...


# ─────────────────────────────────────────────────────────────────
# 2. DataPreparationProvider
# ─────────────────────────────────────────────────────────────────

class DataPreparationProvider(ABC):
    """Abstract interface for data preparation."""

    @abstractmethod
    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """Profile the DataFrame and return a ProfileResult."""
        ...

    @abstractmethod
    def assess_readiness(self, df: pd.DataFrame, target_column: str) -> ReadinessReport:
        """Assess data readiness for ML and return a ReadinessReport."""
        ...

    @abstractmethod
    def detect_leakage(self, df: pd.DataFrame, target_column: str) -> list[LeakageWarning]:
        """Detect data leakage and return a list of LeakageWarnings."""
        ...

    @abstractmethod
    def prepare(self, df: pd.DataFrame, config: dict[str, Any]) -> tuple[Any, Any]:
        """Transform *df* according to *config*. Returns (transformed_df, pipeline)."""
        ...

    @abstractmethod
    def export_pipeline(self, pipeline: Any, path: str) -> str:
        """Serialize *pipeline* to *path* and return the absolute file path."""
        ...


# ─────────────────────────────────────────────────────────────────
# 3. AIProvider
# ─────────────────────────────────────────────────────────────────

class AIProvider(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion for *prompt*."""
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Generate a structured JSON response conforming to *schema*."""
        ...

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        """Estimate the cost in USD for the given token counts."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the list of models available from this provider."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the API key is valid."""
        ...


# ─────────────────────────────────────────────────────────────────
# 4. ExperimentRunner
# ─────────────────────────────────────────────────────────────────

class ExperimentRunner(ABC):
    """Abstract interface for running ML experiments."""

    @abstractmethod
    async def run(self, spec: Any) -> Any:
        """Execute an experiment from *spec* and return an ExperimentResult."""
        ...

    @abstractmethod
    def validate_spec(self, spec: Any) -> list[str]:
        """Validate the experiment spec. Return a list of error strings (empty = valid)."""
        ...

    @abstractmethod
    async def get_status(self, experiment_id: str) -> str:
        """Return the current status string for the experiment."""
        ...

    @abstractmethod
    async def cancel(self, experiment_id: str) -> bool:
        """Cancel a running experiment. Return True if successful."""
        ...


# ─────────────────────────────────────────────────────────────────
# 5. ModelProvider
# ─────────────────────────────────────────────────────────────────

class ModelProvider(ABC):
    """Abstract interface for ML model lifecycle management."""

    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: Any,
        params: dict[str, Any],
    ) -> Any:
        """Train a model. Return the trained model object."""
        ...

    @abstractmethod
    def predict(self, model: Any, X: Any) -> Any:
        """Generate predictions for *X* using *model*."""
        ...

    @abstractmethod
    def save(self, model: Any, path: str) -> str:
        """Serialize *model* to *path*. Return the absolute file path."""
        ...

    @abstractmethod
    def load(self, path: str) -> Any:
        """Deserialize a model from *path*."""
        ...

    @abstractmethod
    def get_feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Return a {feature_name: importance_score} mapping."""
        ...


# ─────────────────────────────────────────────────────────────────
# 6. MetricProvider
# ─────────────────────────────────────────────────────────────────

class MetricProvider(ABC):
    """Abstract interface for computing evaluation metrics."""

    @abstractmethod
    def compute_classification(
        self,
        y_true: Any,
        y_pred: Any,
        y_prob: Optional[Any] = None,
    ) -> ClassificationMetrics:
        """Compute classification metrics."""
        ...

    @abstractmethod
    def compute_regression(
        self,
        y_true: Any,
        y_pred: Any,
    ) -> RegressionMetrics:
        """Compute regression metrics."""
        ...

    @abstractmethod
    def compare(
        self,
        baseline: dict[str, float],
        challenger: dict[str, float],
        primary_metric: str,
    ) -> dict[str, Any]:
        """Compare baseline vs challenger metrics. Return comparison summary."""
        ...


# ─────────────────────────────────────────────────────────────────
# 7. ValidationStrategy
# ─────────────────────────────────────────────────────────────────

class ValidationStrategy(ABC):
    """Abstract interface for train/validation splitting strategies."""

    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str) -> list[ValidationSplit]:
        """Return a list of ValidationSplit objects for the given DataFrame."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this validation strategy."""
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return the strategy configuration as a dict."""
        ...


# ─────────────────────────────────────────────────────────────────
# 8. ReportProvider
# ─────────────────────────────────────────────────────────────────

class ReportProvider(ABC):
    """Abstract interface for generating experiment/project reports."""

    @abstractmethod
    async def generate(
        self,
        project_id: str,
        experiment_ids: list[str],
        format: str = "markdown",
        **kwargs: Any,
    ) -> str:
        """Generate and return a report string in the requested *format*."""
        ...


# ─────────────────────────────────────────────────────────────────
# 9. DeploymentProvider
# ─────────────────────────────────────────────────────────────────

class DeploymentProvider(ABC):
    """Abstract interface for model deployment."""

    @abstractmethod
    async def deploy(
        self,
        model_path: str,
        project_id: str,
        config: dict[str, Any],
    ) -> str:
        """Deploy a model. Return a deployment ID."""
        ...

    @abstractmethod
    async def get_endpoint(self, deployment_id: str) -> str:
        """Return the inference endpoint URL for *deployment_id*."""
        ...

    @abstractmethod
    async def get_status(self, deployment_id: str) -> str:
        """Return the deployment status string."""
        ...

    @abstractmethod
    async def teardown(self, deployment_id: str) -> bool:
        """Tear down a deployment. Return True if successful."""
        ...


__all__ = [
    "ProfileResult",
    "LeakageWarning",
    "ReadinessReport",
    "ModelInfo",
    "ClassificationMetrics",
    "RegressionMetrics",
    "ValidationSplit",
    "DatasetProvider",
    "DataPreparationProvider",
    "AIProvider",
    "ExperimentRunner",
    "ModelProvider",
    "MetricProvider",
    "ValidationStrategy",
    "ReportProvider",
    "DeploymentProvider",
]
