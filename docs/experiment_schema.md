# Experiment Schema

The `ExperimentSpec` and `ExperimentResult` Pydantic models define the contract for all experiments in MLPilot.

## ExperimentSpec

```json
{
  "id": "exp-001",
  "parent_id": "exp-000",
  "project_id": "proj-abc",
  "dataset_version": "v2",
  "hypothesis": "Adding polynomial features for 'income' will improve F1 score.",
  "change_description": "Add PolynomialFeatures(degree=2) for income and age columns.",
  "model_name": "XGBClassifier",
  "parameters": {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05
  },
  "validation_config": {
    "strategy": "stratified_kfold",
    "n_splits": 5
  },
  "feature_set": ["age", "income", "score", "category"],
  "preprocessing_config": {
    "imputer": "median",
    "scaler": "standard",
    "encoder": "onehot"
  },
  "budget": {
    "max_runtime_seconds": 300,
    "max_cost_usd": 0.10
  }
}
```

## ExperimentResult

```json
{
  "id": "exp-001",
  "parent_id": "exp-000",
  "project_id": "proj-abc",
  "dataset_version": "v2",
  "hypothesis": "Adding polynomial features for 'income' will improve F1 score.",
  "change_description": "Add PolynomialFeatures(degree=2) for income and age columns.",
  "model_name": "XGBClassifier",
  "parameters": {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05
  },
  "validation_config": {"strategy": "stratified_kfold", "n_splits": 5},
  "metrics": {
    "f1": 0.847,
    "accuracy": 0.863,
    "precision": 0.851,
    "recall": 0.843,
    "roc_auc": 0.921
  },
  "artifacts": ["artifacts/proj-abc/exp-001/model.joblib"],
  "runtime_seconds": 42.3,
  "cost_usd": 0.003,
  "status": "evaluated",
  "decision": "keep",
  "decision_reason": "F1 improved by +2.7% over baseline (exp-000). Recall improved. No regression on precision.",
  "agent_model": "llama-3.3-70b-versatile",
  "timestamp": "2026-09-23T01:00:00Z",
  "parent_metrics": {
    "f1": 0.820,
    "accuracy": 0.841,
    "roc_auc": 0.897
  }
}
```

## Status Lifecycle

```
created → validated → queued → running → completed → evaluated
                                       ↘ failed
```

## Decision Values

| Decision | Meaning |
|----------|-----------------------------------|
| `keep` | Experiment improved — promote |
| `reject` | No improvement or regression |
| `inconclusive` | Mixed results, needs more data |
| `pending` | Not yet decided |
