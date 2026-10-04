# Implementation Notes & Suggestions

*A living document of potential improvements, edge cases, and feature ideas noticed during implementation. Not blocking the current checklist, but worth reviewing before a 1.0 release.*

## 1. Immutable Dataset Versioning
**Current state:** We reference datasets by file path. If a user modifies or overwrites the original CSV on disk, the experiment history becomes fundamentally disconnected from the data it was trained on. 
**Suggestion:** When a file is uploaded or connected, copy it to an internal immutable storage directory (e.g., `backend/data/versions/{sha256_hash}.csv`). Treat this internal copy as read-only. All experiments should reference the hash, guaranteeing perfect reproducibility even if the user deletes their original file.

## 2. Expanded Safe AST Evaluator
**Current state:** The `_safe_eval` function in `executor.py` (and now `ui.py`) successfully blocks malicious code execution, but it's very strict. It only supports `+`, `-`, `*`, `/`, and `**`.
**Suggestion:** Feature engineering frequently requires comparisons (`>`, `<`), boolean logic (`&`, `|`), and basic math functions (`log`, `sqrt`). We should expand the AST whitelist to include `ast.Compare`, `ast.BoolOp`, and safely map specific `ast.Call` nodes to explicitly allowed NumPy/Pandas functions. Otherwise, the agent will propose valid features (like "is_adult = age >= 18") that our executor will crash on.

## 3. Native Prediction Endpoint
**Current state:** We export the final model via `joblib.dump()`, and give the user a Python script to run it. 
**Suggestion:** It would be highly valuable to add a single dynamic FastAPI endpoint (`POST /api/v1/predict/{experiment_id}`) that loads the saved `joblib` pipeline into memory and scores JSON payloads on the fly. This instantly turns the MLPilot output into a microservice, drastically reducing time-to-value for backend developers testing the model.

## 4. Explicit Fallback Logging
**Current state:** In `DecisionAgent`, if the LLM fails to output valid JSON for a keep/reject decision, we catch the exception and fall back to a hardcoded rule (`keep` if F1 improved). 
**Suggestion:** We should add a flag to the `ExperimentResult` schema (e.g., `decision_mode: "agent" | "fallback"`) so the UI can warn the user that the AI didn't actually reason about this specific decision due to an API timeout or parsing error.

## 5. UI: Experiment Graph Pruning
**Current state:** The experiment graph will show every accepted, rejected, and failed run. 
**Suggestion:** If an agent runs a 20-step budget and rejects 15 ideas, the graph will become incredibly wide/noisy. We should implement a "Champion Path Only" toggle in the frontend that hides rejected branches and only shows the lineage of the best model.
