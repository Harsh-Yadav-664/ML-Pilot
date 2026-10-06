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

## Run manifest

When an experiment completes, MLPilot stores a **run manifest** on it (`experiments.manifest`, returned as `manifest` by `GET /projects/{p}/experiments/{id}`). It holds what is needed to reproduce the run and nothing else. It is built in `backend/ml/experiments/manifest.py` from named fields only, so it cannot pick up a secret that happens to sit in the run's parameters.

| Field | What it holds |
|---|---|
| `manifest_version`, `experiment_id`, `parent_id`, `created_at` | identity and format version |
| `mlpilot_version`, `git_sha`, `python_version`, `packages` | the software (`git_sha` is null outside a git checkout; `packages` lists numpy, pandas, scikit-learn, scipy, lightgbm, xgboost, optuna, sqlglot, pydantic) |
| `data_version_id` | SHA-256 of the exact data file the run trained on |
| `task` | target column, class list and the positive class |
| `split_plan`, `seeds` | the split definition (strategy, fractions, folds, seed) and the seeds used |
| `engine` | name, version, the parameters of the final model (given plus tuned) and how tuning was done |
| `preprocessing_config` | the preprocessing override, if any |
| `features` | engineered features in the order they were added: `name`, `kind`, `formula` |
| `acceptance_rule`, `acceptance` | the keep/reject rule and its measured gain, for candidate runs |
| `calibration`, `threshold` | fitted on validation only (see the business metrics) |
| `metrics` | `{"val": {...}, "test": {...}}` |
| `llm` | per LLM call: provider, model, tokens, cost, `decision_mode`, how many providers failed first. No prompt or response text. |
| `timings` | runtime in seconds |

### Replay

`ExperimentService.replay(experiment_id)` retrains and re-scores a completed experiment from its manifest alone, with no LLM call and no tuning:

- it loads the data version named in the manifest and checks its content hash;
- it applies every recorded feature in order as an accepted feature;
- it trains the recorded engine with the recorded parameters, split and seed.

On the same machine and package versions the replayed validation and test metrics match the recorded ones to 1e-9. Bit-exact results across different operating systems or CPUs are not promised.
