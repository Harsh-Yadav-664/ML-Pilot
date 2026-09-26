# MLPilot Provider Interfaces

All 9 ABCs are defined in `backend/ml/core/interfaces.py`.

## 1. DatasetProvider

Loads datasets from files.

```python
class DatasetProvider(ABC):
    def load(self, path: str, **kwargs) -> pd.DataFrame: ...
    def supports(self, path: str) -> bool: ...
    def schema(self, df: pd.DataFrame) -> dict[str, str]: ...
```

**Implementations**: `CsvLoader`, `ParquetLoader`

## 2. DataPreparationProvider

Profiles, assesses, and prepares data for ML.

```python
class DataPreparationProvider(ABC):
    def profile(self, df) -> ProfileResult: ...
    def assess_readiness(self, df, target_column) -> ReadinessReport: ...
    def detect_leakage(self, df, target_column) -> list[LeakageWarning]: ...
    def prepare(self, df, config) -> tuple[df, pipeline]: ...
    def export_pipeline(self, pipeline, path) -> str: ...
```

**Implementations**: `NativeDataPreparationProvider` (Phase 1), `DataCleanAdapter` (future)

## 3. AIProvider

Abstracts language model providers.

```python
class AIProvider(ABC):
    async def complete(self, prompt, system, model, max_tokens, temperature) -> str: ...
    async def complete_structured(self, prompt, schema, ...) -> dict: ...
    def estimate_cost(self, prompt_tokens, completion_tokens, model) -> float: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def health_check(self) -> bool: ...
```

**Implementations**: `StubProvider`, `GroqProvider`, `GeminiProvider`, `NvidiaNIMProvider`, `OpenRouterProvider`, `CerebrasProvider`, `MistralProvider`, `OpenAIProvider`, `AnthropicProvider`

## 4. ExperimentRunner

Executes ML experiments from spec to result.

```python
class ExperimentRunner(ABC):
    async def run(self, spec: ExperimentSpec) -> ExperimentResult: ...
    def validate_spec(self, spec) -> list[str]: ...
    async def get_status(self, experiment_id) -> str: ...
    async def cancel(self, experiment_id) -> bool: ...
```

**Implementations**: `LocalExperimentExecutor`

## 5. ModelProvider

Manages ML model lifecycle.

```python
class ModelProvider(ABC):
    def train(self, X_train, y_train, params) -> model: ...
    def predict(self, model, X) -> predictions: ...
    def save(self, model, path) -> str: ...
    def load(self, path) -> model: ...
    def get_feature_importance(self, model, feature_names) -> dict: ...
```

**Implementations**: `JobLibModelProvider`

## 6. MetricProvider

Computes evaluation metrics.

```python
class MetricProvider(ABC):
    def compute_classification(self, y_true, y_pred, y_prob) -> ClassificationMetrics: ...
    def compute_regression(self, y_true, y_pred) -> RegressionMetrics: ...
    def compare(self, baseline, challenger, primary_metric) -> dict: ...
```

**Implementations**: `SklearnClassificationMetricProvider`, `SklearnRegressionMetricProvider`

## 7. ValidationStrategy

Defines train/validation splits.

```python
class ValidationStrategy(ABC):
    def split(self, df, target_column) -> list[ValidationSplit]: ...
    def describe(self) -> str: ...
    def get_config(self) -> dict: ...
```

**Implementations**: `StratifiedKFoldStrategy`, `KFoldStrategy`, `TimeSeriesSplitStrategy`, `HoldoutStrategy`

## 8. ReportProvider

Generates experiment and project reports.

```python
class ReportProvider(ABC):
    async def generate(self, project_id, experiment_ids, format) -> str: ...
```

## 9. DeploymentProvider

Deploys trained models to serving infrastructure.

```python
class DeploymentProvider(ABC):
    async def deploy(self, model_path, project_id, config) -> str: ...
    async def get_endpoint(self, deployment_id) -> str: ...
    async def get_status(self, deployment_id) -> str: ...
    async def teardown(self, deployment_id) -> bool: ...
```
