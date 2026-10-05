# MLPilot roadmap issues (draft)

Repository: Harsh-Yadav-664/ML-Pilot. 50 issues + 1 tracking issue.


## Phase 0

- **[0.1] Backend does not install from a fresh clone (UTF-16 line in requirements, missing greenlet)** (P0, phase-0, area:backend, area:ci, type:bug)
- **[0.2] Backend crashes on start: chat.py imports a missing module and calls gateway methods that don't exist** (P0, phase-0, area:backend, area:llm, type:bug)
- **[0.3] /ui/agent/suggestions silently returns [] (calls removed planner method); stale planner test** (P0, phase-0, area:backend, area:llm, type:bug)
- **[0.4] Bundled telecom sample fails: classification targets are never label-encoded** (P0, phase-0, area:ml, type:bug)
- **[0.5] Baseline experiment stuck in 'created' (background task runs before the row is committed)** (P0, phase-0, area:backend, type:bug)
- **[0.6] Frontend: delete duplicate UI, fix API prefix, replace silent mock fallback with an explicit Demo mode** (P0, phase-0, area:frontend, type:bug)
- **[0.7] Frontend: wire upload, sample data, experiments, agent loop and chat to the real backend; remove simulated data** (P0, phase-0, area:frontend, area:backend, type:bug)
- **[0.8] Remove fake SHAP values from the debrief endpoint** (P0, phase-0, area:backend, area:ml, type:bug)
- **[0.9] Security quick fixes: arbitrary file read via dataset_path, weak SQL read-only check** (P0, phase-0, area:security, area:backend, type:bug)
- **[0.10] Add CI: clean install, unit tests, frontend build, end-to-end sample run** (P0, phase-0, area:ci, type:feature)
- **[0.11] README tells the truth: problem, solution, what works today, how to run** (P0, phase-0, area:docs, type:docs)
- **[0.12] Repo hygiene: close stale PR #13, add LICENSE, remove dead code and stray data** (P1, phase-0, area:docs, type:chore)

## Phase 1

- **[1.1] Stop tuning on the test set: train/validation/test split with nested tuning** (P1, phase-1, area:ml, type:bug)
- **[1.2] Accept a feature only when the gain beats noise; keep/reject decided by code, not the LLM** (P1, phase-1, area:ml, area:llm, type:feature)
- **[1.3] Make accepted features cumulative; the final model must be the thing that was measured** (P1, phase-1, area:ml, type:feature)
- **[1.4] Safe evaluator: reject invalid formulas (no silent zeros), support comparisons and whitelisted functions** (P1, phase-1, area:ml, area:security, type:feature)
- **[1.5] Rewrite leakage detector: fix false positives, add single-feature predictiveness checks, measure on planted leaks** (P1, phase-1, area:leakage, area:ml, type:feature)
- **[1.6] Immutable dataset versions by content hash** (P2, phase-1, area:backend, area:data-connect, type:feature)
- **[1.7] Model engine abstraction: LightGBM default, optional AutoGluon and TabICL** (P2, phase-1, area:ml, type:feature)
- **[1.8] LLM routing: config-driven, all providers routable, record fallbacks, add local Ollama provider** (P2, phase-1, area:llm, type:feature)

## Phase 2

- **[2.1] DuckDB as the single internal engine (CSV/Parquet become one-table databases)** (P1, phase-2, area:data-connect, area:backend, type:feature)
- **[2.2] Connection manager: Postgres, SQLite, DuckDB connections with safe secret handling** (P1, phase-2, area:data-connect, area:security, area:backend, type:feature)
- **[2.3] Read-only enforcement in depth (parser allowlist + read-only transaction + privilege check + limits)** (P1, phase-2, area:security, area:data-connect, type:feature)
- **[2.4] Schema introspection and relationship graph (declared + inferred keys, time columns)** (P1, phase-2, area:data-connect, type:feature)
- **[2.5] UI: 'Connect a database' flow with schema graph view** (P1, phase-2, area:frontend, type:feature)
- **[2.6] Demo relational database (Postgres in docker-compose) with a realistic schema and planted leaks** (P1, phase-2, area:data-connect, area:ci, type:feature)
- **[2.7] Privacy controls: what the LLM may see, with an auditable prompt log** (P2, phase-2, area:security, area:llm, type:feature)

## Phase 3

- **[3.1] Prediction task spec: schema, validation and storage** (P1, phase-3, area:ml, area:leakage, type:feature)
- **[3.2] Deterministic training-table builder: labels at cutoff dates** (P1, phase-3, area:ml, area:leakage, area:data-connect, type:feature)
- **[3.3] Point-in-time guard: SQL rewriter/validator that forces every feature query to respect the cutoff** (P1, phase-3, area:leakage, area:security, type:feature)
- **[3.4] Temporal validation for relational tasks (split by cutoff date, not at random)** (P1, phase-3, area:ml, area:leakage, type:feature)
- **[3.5] Natural-language question -> task spec, confirmed by the user in chat** (P2, phase-3, area:llm, area:frontend, type:feature)
- **[3.6] Leakage canaries: planted future data must be blocked, proven in CI** (P1, phase-3, area:leakage, area:ci, type:feature)

## Phase 4

- **[4.1] Deterministic baseline features across related tables (DFS-style, cutoff-aware)** (P1, phase-4, area:ml, area:data-connect, type:feature)
- **[4.2] LLM SQL feature proposer (one feature at a time, history-aware, structured output)** (P1, phase-4, area:llm, area:ml, type:feature)
- **[4.3] Feature execution engine: batch over all cutoffs, cache, time and cost budget** (P2, phase-4, area:data-connect, area:ml, type:feature)
- **[4.4] Agent loop for relational tasks (replace formula loop; champion path; optional N rollouts)** (P2, phase-4, area:llm, area:ml, type:feature)
- **[4.5] Chat human-in-the-loop: checkpoints, live narration, settings that actually change the run** (P2, phase-4, area:llm, area:frontend, area:backend, type:feature)

## Phase 5

- **[5.1] Evidence record per decision and a run report (HTML/Markdown)** (P2, phase-5, area:ml, area:docs, type:feature)
- **[5.2] Export bundle: SQL/dbt models + model file + scoring script + task spec, reproducible** (P2, phase-5, area:ml, area:data-connect, type:feature)
- **[5.3] Batch scoring: predict for the latest cutoff from the UI/CLI** (P2, phase-5, area:ml, area:backend, type:feature)
- **[5.4] Real SHAP explanations, grounded debrief, and grounded Q&A over run history** (P2, phase-5, area:ml, area:llm, type:feature)
- **[5.5] Optional MLflow logging of runs** (P3, phase-5, area:ml, type:feature)

## Phase 6

- **[6.1] RelBench harness: run MLPilot on rel-f1 with the official evaluator** (P2, phase-6, area:benchmark, type:research)
- **[6.2] Publish the benchmark table (with losses) in the README** (P2, phase-6, area:benchmark, area:docs, type:docs)
- **[6.3] Single-table sanity benchmark vs AutoGluon on OpenML (backup-floor check)** (P3, phase-6, area:benchmark, type:research)

## Phase 7

- **[7.1] Package: `pip install mlpilot` CLI and one-command Docker** (P2, phase-7, area:ci, type:feature)
- **[7.2] MCP server so AI assistants can use MLPilot as a tool** (P3, phase-7, area:llm, type:feature)
- **[7.3] Docs site, 2-minute demo video and hosted demo** (P3, phase-7, area:docs, type:docs)
- **[7.4] Later: write predictions back, scheduled re-scoring, more connectors** (P3, phase-7, area:data-connect, type:feature)

---

## [0.1] Backend does not install from a fresh clone (UTF-16 line in requirements, missing greenlet)

Labels: P0, phase-0, area:backend, area:ci, type:bug

## Context

`pip install -r backend/requirements.txt` fails on a clean machine. The last line (`optuna>=3.0.0`) is saved as UTF-16 with null bytes, so pip reports `Invalid requirement: 'o\x00p\x00t\x00u\x00n\x00a...'`. With that fixed, the server still fails because SQLAlchemy async needs `greenlet`, which isn't listed.

## Files

- `backend/requirements.txt`
- `backend/pyproject.toml`

## What to do

- Re-save `backend/requirements.txt` as plain UTF-8 (no BOM, no null bytes).
- Add `greenlet` (or use `sqlalchemy[asyncio]`).
- Pin lower bounds for every dependency; keep `requirements.txt` and `pyproject.toml` consistent (one should be generated from the other, or document which is the source of truth).
- Add a `.gitattributes` rule `*.txt text eol=lf` so the encoding doesn't regress.

## Acceptance criteria

- [ ] In a fresh `python -m venv` on Python 3.13 (the version `backend/pyproject.toml` requires), `pip install -r backend/requirements.txt` succeeds (paste the output tail).
- [ ] `file backend/requirements.txt` reports ASCII/UTF-8 text.
- [ ] `python -c "import sqlalchemy.ext.asyncio, greenlet"` succeeds in that venv.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Honest review, section 2 (project files: reviews/mlpilot-honest-review.md)

## Depends on

nothing



Priority **P0** · Phase **0**

---

## [0.2] Backend crashes on start: chat.py imports a missing module and calls gateway methods that don't exist

Labels: P0, phase-0, area:backend, area:llm, type:bug

## Context

`backend/app/api/v1/chat.py:11` imports `from ai.prompts.tasks import TaskType`; that module was never committed (`TaskType` lives in `ai/router.py`), so `uvicorn app.main:app` fails on a fresh clone. Even after fixing the import, `chat.py:70` and `chat.py:120` call `gateway.route_request(...)` and `ml/agents/cleaning_agent.py:60` calls `gateway.chat_completion(...)`. Neither method exists on `AIGateway` (it has `complete` and `complete_structured`, `backend/ai/gateway.py:102,162`). Chat therefore errors, and Auto-Clean catches the exception and silently uses an empty cleaning recipe.

## Files

- `backend/app/api/v1/chat.py`
- `backend/ml/agents/cleaning_agent.py`
- `backend/ai/gateway.py`
- `backend/ai/router.py`

## What to do

- Import `TaskType` from `ai.router`.
- Replace `route_request` / `chat_completion` calls with `AIGateway.complete` or `complete_structured` (use structured output with a pydantic schema where JSON is expected, instead of stripping ``` fences by hand).
- When the LLM call fails, the cleaning agent must return an explicit error/`decision_mode='fallback'` that the API surfaces, not an empty recipe that looks like success.
- Add unit tests that call each chat endpoint and the cleaning agent with the stub provider.

## Acceptance criteria

- [ ] From a fresh clone (after [0.1]), `cd backend && uvicorn app.main:app` starts with no ImportError.
- [ ] `grep -rn "route_request\|chat_completion" backend/` returns nothing.
- [ ] New tests for `/chat/ask`, `/chat/debrief`, `/chat/settings` and the cleaning agent pass with the stub provider.
- [ ] Forcing the provider to raise makes Auto-Clean return a visible error, covered by a test.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.1]



Priority **P0** · Phase **0**

---

## [0.3] /ui/agent/suggestions silently returns [] (calls removed planner method); stale planner test

Labels: P0, phase-0, area:backend, area:llm, type:bug

## Context

`backend/app/services/experiment_service.py:139` calls `planner.generate_hypotheses(...)`, which was removed when the planner became sequential (`ml/experiments/planner.py:19` now has `generate_next_hypothesis`). The exception is swallowed, so `GET /api/v1/ui/agent/suggestions` returns `[]`. `backend/tests/unit/test_planner.py:33` calls the same removed method and is the one failing unit test.

## Files

- `backend/app/services/experiment_service.py`
- `backend/ml/experiments/planner.py`
- `backend/tests/unit/test_planner.py`
- `backend/app/api/v1/ui.py`

## What to do

- Use `generate_next_hypothesis` (called N times with growing history, or return one suggestion) in the service.
- Stop swallowing exceptions in this path: log and return an HTTP error with a clear message.
- Rewrite `test_planner.py` for the current API.

## Acceptance criteria

- [ ] `cd backend && pytest` passes with 0 failures.
- [ ] `GET /api/v1/ui/agent/suggestions?dataset_path=<sample>&target_column=Churn` with the stub provider returns at least one suggestion (integration test).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2]



Priority **P0** · Phase **0**

---

## [0.4] Bundled telecom sample fails: classification targets are never label-encoded

Labels: P0, phase-0, area:ml, type:bug

## Context

Running the agent loop on `backend/datasets/telecom_churn.csv` (target `Churn` with values `Yes`/`No`) fails with XGBoost: `Expected: [0 1], got ['No' 'Yes']`. The bundled demo dataset can't complete.

## Files

- `backend/ml/experiments/executor.py`
- `backend/ml/baselines/classification.py`
- `backend/ml/data/preparation/native/native_prep.py`

## What to do

- Encode classification targets with `LabelEncoder` (fit on training labels only) in one shared place used by baseline and experiments.
- Store the class mapping with the experiment so predictions and exports decode back to original labels.
- Pick the positive class deterministically for binary metrics (configurable, default = minority class) and record it.

## Acceptance criteria

- [ ] Integration test: baseline + 2 agent iterations on `telecom_churn.csv` complete with the stub provider.
- [ ] Exported script predicts `Yes`/`No`, not `0`/`1`.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2]



Priority **P0** · Phase **0**

---

## [0.5] Baseline experiment stuck in 'created' (background task runs before the row is committed)

Labels: P0, phase-0, area:backend, type:bug

## Context

Starting a baseline on the telecom sample leaves it in `created` forever; the log shows `Experiment ... not found for execution`. The background task starts before the experiment row is visible to its DB session.

## Files

- backend/app/api/v1/ui.py (`/experiments/baseline`, `/experiments/run`)
- `backend/app/services/experiment_service.py`
- `backend/app/db/session.py`

## What to do

- Commit the experiment row before scheduling the background task, and give the task its own session.
- If an experiment can't be found or fails, set status `failed` with the error message instead of leaving it in `created`.

## Acceptance criteria

- [ ] Integration test: `POST /ui/experiments/baseline` followed by polling reaches `completed` (or `failed` with a message) within the test timeout, 20 runs in a row.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2]



Priority **P0** · Phase **0**

---

## [0.6] Frontend: delete duplicate UI, fix API prefix, replace silent mock fallback with an explicit Demo mode

Labels: P0, phase-0, area:frontend, type:bug

## Context

`frontend/src` (started by `start.py`) and `frontend/new_ui/src` are near-identical copies. `frontend/src/api/index.ts:19` uses `http://localhost:8000/api/v1`, but the routes live under `/api/v1/ui/...`, so every call 404s and `withFallback` (`index.ts:29`) quietly returns mock data. Users see fabricated numbers without knowing it.

## Files

- `frontend/src/api/index.ts`
- `frontend/src/api/mock.ts`
- `frontend/new_ui/`
- `start.py`
- `frontend/src/components/Chrome.tsx`

## What to do

- Delete `frontend/new_ui/`; keep `frontend/` (the one `start.py` runs).
- Read the API base URL from `VITE_API_BASE_URL` with default `http://localhost:8000/api/v1/ui`.
- Remove `withFallback`. On API failure show an error state. Mock data is only allowed when the user explicitly turns on **Demo mode**, and a persistent banner says 'Demo mode: sample data, not connected'.

## Acceptance criteria

- [ ] `frontend/new_ui` no longer exists; `npm ci && npm run build` in `frontend/` succeeds.
- [ ] With the backend stopped, the UI shows an error/disconnected state, not numbers.
- [ ] `grep -rn withFallback frontend/src` returns nothing.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

nothing



Priority **P0** · Phase **0**

---

## [0.7] Frontend: wire upload, sample data, experiments, agent loop and chat to the real backend; remove simulated data

Labels: P0, phase-0, area:frontend, area:backend, type:bug

## Context

Upload is parsed in the browser and column stats are `Math.random()` (`frontend/src/store.tsx:523-538`). Experiment metrics come from `simulateMetrics()` (`store.tsx:141`, `:362`) and the real `runExperiment` result is discarded. Chat replies come from `craftReply` in `api/mock.ts`; the real `/chat/*` endpoints are never called. `views/Deployments.tsx` and drift charts show pure mock data (`mockDrift`, `mockEndpoints`, `mockVolume`).

## Files

- `frontend/src/store.tsx`
- `frontend/src/api/index.ts`
- `frontend/src/api/mock.ts`
- `frontend/src/views/*.tsx`
- `backend/app/api/v1/ui.py`
- `backend/app/api/v1/chat.py`

## What to do

- Upload sends the file to `POST /ui/data/upload` and renders stats returned by `/ui/data/metrics`.
- 'Try sample data' calls `POST /ui/data/sample`.
- Baseline, experiments and auto-optimize use the backend and poll real status/metrics.
- Chat panel calls the real `/chat/*` endpoints (after [0.2]).
- Remove `simulateMetrics`, `Math.random()` stats and `craftReply` from the non-demo path.
- Hide the Deployments view and drift charts, or label them 'Planned' with no numbers.

## Acceptance criteria

- [ ] `grep -n "Math.random\|simulateMetrics\|craftReply" frontend/src` only matches code behind the Demo-mode flag.
- [ ] Manual run recorded in the PR (screenshot or short video): fresh clone, `python start.py`, sample data, baseline, 2 agent iterations, chat question; the numbers match `GET /ui/experiments/tree`.
- [ ] A Playwright smoke test runs the same flow against the stub provider (can be added in [0.10]).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2], [0.3], [0.4], [0.5], [0.6]



Priority **P0** · Phase **0**

---

## [0.8] Remove fake SHAP values from the debrief endpoint

Labels: P0, phase-0, area:backend, area:ml, type:bug

## Context

`backend/app/api/v1/chat.py:113-121` sends hardcoded values (`{"age": 0.35, "balance": 0.22, "is_active": -0.15}`) to the LLM as 'SHAP feature importance' and returns them to the user, whatever the dataset. `shap` isn't in the requirements. This produces confident explanations of columns that may not exist.

## Files

- `backend/app/api/v1/chat.py`

## What to do

- Remove the mock values now. Until real SHAP exists ([5.4]), the debrief uses only the recorded metrics and the model's built-in feature importances for the real columns, and says that it does.

## Acceptance criteria

- [ ] `grep -rn mock_shap backend/` returns nothing.
- [ ] Test: the debrief for an experiment on the telecom sample only mentions columns that exist in that dataset.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2]



Priority **P0** · Phase **0**

---

## [0.9] Security quick fixes: arbitrary file read via dataset_path, weak SQL read-only check

Labels: P0, phase-0, area:security, area:backend, type:bug

## Context

Several endpoints in `backend/app/api/v1/ui.py` take `dataset_path` straight from the query string or body (lines 122, 144, 170, 206, 231, 293, 317) and pass it to the loader, so any readable file on the server can be loaded. `POST /ui/data/connect-sql` (`ui.py:96`) accepts any connection string and query; `ml/data/ingestion/sql_loader.py` only checks `startswith("select"|"with")`, which `WITH x AS (DELETE ...) SELECT ...` passes in Postgres, and multi-statement strings aren't blocked. Error messages echo exceptions that can include the connection string.

## Files

- `backend/app/api/v1/ui.py`
- `backend/ml/data/ingestion/sql_loader.py`

## What to do

- Replace `dataset_path` with a dataset ID that maps to a file under the uploads/datasets directory; reject anything else (resolve the path and check it's inside the allowed dir).
- Interim SQL hardening until Phase 2: parse with `sqlglot`, allow exactly one `SELECT` statement with no data-modifying CTEs, run inside a read-only transaction, apply a statement timeout and a row limit.
- Never include the connection string or password in logs or error responses.

## Acceptance criteria

- [ ] Tests: `dataset_path=../../etc/passwd` (or the ID equivalent) is rejected with 400.
- [ ] Tests: `INSERT`, `UPDATE`, `DROP`, `WITH x AS (DELETE ...) SELECT 1`, and `SELECT 1; DROP TABLE t` are all refused against a SQLite fixture.
- [ ] Test: a failed connection's error response doesn't contain the password.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Full read-only design comes in [2.3]

## Depends on

[0.2]



Priority **P0** · Phase **0**

---

## [0.10] Add CI: clean install, unit tests, frontend build, end-to-end sample run

Labels: P0, phase-0, area:ci, type:feature

## Context

There's no `.github/workflows`. Nothing caught the encoding bug, the missing module, or the broken suggestions endpoint.

## Files

- .github/workflows/ci.yml (new)
- backend/tests/integration/ (new tests)

## What to do

- Workflow on push and pull_request: Python 3.13 (the version `backend/pyproject.toml` requires), install `backend/requirements.txt`, run `ruff` (or flake8) and `pytest`.
- Frontend job: `npm ci && npm run build` (and `tsc --noEmit`).
- End-to-end test with the stub provider: load `telecom_churn.csv`, run baseline + 2 agent iterations, assert all experiments reach `completed` and metrics are numbers.
- Optional: Playwright smoke test of the UI flow from [0.7].

## Acceptance criteria

- [ ] CI is green on the PR that adds it, and red on a test branch that reintroduces the UTF-16 line (link both runs in the PR).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.1], [0.2], [0.3], [0.4], [0.5]



Priority **P0** · Phase **0**

---

## [0.11] README tells the truth: problem, solution, what works today, how to run

Labels: P0, phase-0, area:docs, type:docs

## Context

The README contains an interview-prep cheat sheet, lists shipped items as future milestones, and claims features that don't run (see the honest review). The Master Build Checklist section 2 asks for problem -> solution -> what the tool does.

## Files

- `README.md`
- docs/concepts.md (new, optional)

## What to do

- Structure: problem, solution, what works today (only things covered by CI), roadmap (link to the tracking issue), quick start, architecture.
- Move the ML concepts section to `docs/concepts.md` or delete it.
- Say plainly what's planned vs working; e.g. 'direct database connection: planned (Phase 2)' until [2.5] is merged.
- Leakage section states the real number of checks (the code has 7 `detect_*` methods in `ml/validation/leakage.py`, docs say 6).

## Acceptance criteria

- [ ] Every feature listed under 'works today' has a test or CI step that exercises it (list them in the PR).
- [ ] The quick start, followed literally on a fresh machine, works (reviewer confirms).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.7], [0.10]



Priority **P0** · Phase **0**

---

## [0.12] Repo hygiene: close stale PR #13, add LICENSE, remove dead code and stray data

Labels: P1, phase-0, area:docs, type:chore

## Context

PR #13 'Phase 2C' is still open, but its work was merged manually in `60016b3`. There's no LICENSE file. The DataClean adapter is dead code since the decision to go native. `backend/data.csv` is the default `dataset_path` in many endpoints.

## Files

- LICENSE (new)
- `backend/ml/data/preparation/dataclean/`
- `backend/data.csv`
- `suggestions.md`

## What to do

- Close PR #13 with a comment pointing at `60016b3`.
- Add `LICENSE` (proposed default: Apache-2.0; Harsh to confirm).
- Delete `ml/data/preparation/dataclean/` and its references.
- Move `backend/data.csv` to `backend/datasets/` or tests fixtures; no endpoint should default to it.
- Fold the still-relevant items of `suggestions.md` into issues and delete or archive the file.

## Acceptance criteria

- [ ] PR #13 closed; LICENSE present; `grep -rn dataclean backend/` returns nothing; tests pass.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

nothing



Priority **P1** · Phase **0**

---

## [1.1] Stop tuning on the test set: train/validation/test split with nested tuning

Labels: P1, phase-1, area:ml, type:bug

## Context

In `backend/ml/experiments/executor.py`, the Optuna objective fits on `X_train` and scores on `X_test` (around line 188), and the final metric is reported on the same `X_test` (line 217). That's selection bias: reported metrics are optimistic, inside a tool whose pitch is leakage safety.

## Files

- `backend/ml/experiments/executor.py`
- `backend/ml/validation/strategies.py`

## What to do

- Split into train / validation / test (or CV on train for tuning) with a fixed seed recorded in the experiment.
- Optuna tunes on validation (or inner CV) only; refit on train(+val) and evaluate on test once.
- Record which split each metric comes from in the experiment record.

## Acceptance criteria

- [ ] Unit test with a spy: the test split is passed to `predict` exactly once per experiment, after tuning.
- [ ] Experiment records contain `val_*` and `test_*` metrics separately.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Cawley & Talbot, 'On over-fitting in model selection' (JMLR 2010)

## Depends on

[0.10]



Priority **P1** · Phase **1**

---

## [1.2] Accept a feature only when the gain beats noise; keep/reject decided by code, not the LLM

Labels: P1, phase-1, area:ml, area:llm, type:feature

## Context

Today each feature is compared with the baseline on one 80/20 split, any increase counts, and `DecisionAgent` asks the LLM 'baseline F1 X, new F1 Y, keep?'. In the review run, features that were a constant column of zeros were 'kept' because of tuning noise.

## Files

- `backend/ml/agents/decision_agent.py`
- `backend/ml/experiments/executor.py`
- `backend/ml/experiments/schema.py`

## What to do

- Evaluate candidate vs current champion with repeated K-fold CV (default 5x2 or 3x5, same folds for both).
- Accept only if the mean gain > a margin (default: max(0.002, 1 std of the paired fold differences)) or a paired test passes; make the rule configurable and record it.
- Move keep/reject into deterministic code. The LLM only writes the explanation, and the record says `decision_mode: rule`.
- Record per-fold scores and the gain's confidence interval in the experiment.

## Acceptance criteria

- [ ] Test: a constant feature and a pure-noise feature are rejected on the telecom sample in 10/10 seeds.
- [ ] Test: a planted informative feature (e.g. a noisy copy of the target signal) is accepted.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- CAAFE accepts features only on CV improvement: https://arxiv.org/abs/2305.03403
- Dietterich 1998, 5x2cv paired t-test

## Depends on

[1.1]



Priority **P1** · Phase **1**

---

## [1.3] Make accepted features cumulative; the final model must be the thing that was measured

Labels: P1, phase-1, area:ml, type:feature

## Context

Every experiment is baseline + one feature (`parent_id=baseline_id`, `backend/ml/agents/decision_agent.py:112`). Winners are never combined, so the model a user would export was never evaluated.

## Files

- `backend/ml/agents/decision_agent.py`
- `backend/app/services/experiment_service.py`
- `frontend/src/components/ExperimentGraph.tsx`

## What to do

- Keep a 'champion' experiment; each candidate = champion + 1 new feature; an accepted candidate becomes the new champion (`parent_id = champion_id`).
- The export endpoint exports the champion.
- The experiment graph shows the champion path (see `suggestions.md` item 5).

## Acceptance criteria

- [ ] Test: after 3 accepted features, the champion's feature list has all 3 and its metrics come from a run containing all 3.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[1.2]



Priority **P1** · Phase **1**

---

## [1.4] Safe evaluator: reject invalid formulas (no silent zeros), support comparisons and whitelisted functions

Labels: P1, phase-1, area:ml, area:security, type:feature

## Context

`backend/ml/experiments/executor.py:73-102` has a correct whitelist AST evaluator, but when a formula fails it falls back to a column of `0` (line 102). It only supports + - * / ** and unary minus, so common features (`age >= 18`, `log(income)`) fail. `x ** 10**9` can hang a worker. The evaluator is duplicated in `ui.py`.

## Files

- `backend/ml/experiments/executor.py`
- `backend/app/api/v1/ui.py`
- backend/ml/features/safe_eval.py (new)

## What to do

- Move the evaluator into one module used everywhere.
- Invalid formula -> experiment status `rejected_invalid` with the parse error; never a zero column.
- Allow comparisons, `&`/`|`/`~`, and a whitelist of functions mapped to NumPy (`log1p`, `sqrt`, `abs`, `clip`, `where`, `minimum`, `maximum`). Cap exponents and constant sizes.
- Fuzz test with random/hostile ASTs (attribute access, calls to non-whitelisted names, lambdas, comprehensions).

## Acceptance criteria

- [ ] Tests: `__import__('os')`, `().__class__`, `open('x')`, `a ** 10**9` are rejected quickly; `log1p(income) / (age + 1)` and `where(age >= 18, 1, 0)` work.
- [ ] No code path inserts a constant fallback column.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- suggestions.md item 2

## Depends on

[0.10]



Priority **P1** · Phase **1**

---

## [1.5] Rewrite leakage detector: fix false positives, add single-feature predictiveness checks, measure on planted leaks

Labels: P1, phase-1, area:leakage, area:ml, type:feature

## Context

`backend/ml/validation/leakage.py` matches substrings: `TIMESTAMP_KEYWORDS` includes `"dt"` and `"ts"` (line 16), so `Dependents` and `InternetService` are flagged as timestamps; the contamination check looks for `"min"` (line 181), flagging `StreamingTV`. The correlation check (> 0.95, line 57) only runs for numeric targets, so it never runs for Yes/No targets. All 6 warnings on the telecom sample are false positives except `customerID`.

## Files

- `backend/ml/validation/leakage.py`
- `backend/tests/unit/test_leakage.py`
- backend/tests/fixtures/leakage/ (new)

## What to do

- Tokenize column names (snake/camel case) and match whole tokens.
- Add a single-feature check: for each column, cross-validated AUC (classification) or R^2 (regression) of a shallow tree on that column alone; flag suspiciously high values with the score as evidence. Works for categorical targets.
- ID detection by uniqueness ratio + monotonicity, not only by name.
- Map each check to a category from Kapoor & Narayanan's taxonomy and return evidence (score, examples).
- Build 10+ small fixture datasets with planted leaks (target copy, post-outcome column, ID that encodes the label, global normalisation, duplicated rows across split) and clean controls. Report precision/recall in a test and in the README.

## Acceptance criteria

- [ ] On `telecom_churn.csv`, only `customerID` (ID) is flagged.
- [ ] On the planted-leak suite, recall >= 0.9 and precision >= 0.8, printed by CI.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Kapoor & Narayanan, Leakage and the reproducibility crisis in ML-based science: https://arxiv.org/abs/2207.07048
- Yang et al., Data Leakage in Notebooks (ASE 2022): https://arxiv.org/abs/2209.03345

## Depends on

[0.10]



Priority **P1** · Phase **1**

---

## [1.6] Immutable dataset versions by content hash

Labels: P2, phase-1, area:backend, area:data-connect, type:feature

## Context

Experiments reference datasets by file path. If the file changes, history no longer matches the data it was trained on (`suggestions.md` item 1).

## Files

- `backend/app/api/v1/ui.py`
- `backend/app/db/models/dataset.py`
- `backend/ml/data/ingestion/*`

## What to do

- On upload/sample/SQL snapshot, store a read-only copy named by SHA-256 of its content; experiments reference the hash.
- Record row/column counts and the source (upload, sample, SQL query + connection name without secrets).

## Acceptance criteria

- [ ] Test: modifying the original file after upload doesn't change an experiment's re-run result.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.9]



Priority **P2** · Phase **1**

---

## [1.7] Model engine abstraction: LightGBM default, optional AutoGluon and TabICL

Labels: P2, phase-1, area:ml, type:feature

## Context

MLPilot shouldn't compete with AutoGluon on model fitting; it should use strong engines and focus on data, features and validation. LightGBM is fast and permissively licensed; AutoGluon (Apache-2.0) and TabICL v2 (open) are strong optional engines. Note: RealTabPFN-2.5 is non-commercial, so don't make it a default.

## Files

- `backend/ml/models/provider.py`
- `backend/ml/experiments/executor.py`
- `backend/ml/core/registry.py`

## What to do

- Define a `ModelEngine` interface (fit, predict_proba, feature_importances, save/load).
- Implement LightGBM (default) and keep XGBoost; AutoGluon and TabICL as optional extras (`pip install mlpilot[autogluon]`).
- Engine and its version are recorded on every experiment.

## Acceptance criteria

- [ ] The e2e test passes with LightGBM; an optional CI job runs it with AutoGluon `presets='medium_quality'` and a time limit.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://auto.gluon.ai
- https://github.com/soda-inria/tabicl
- https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models

## Depends on

[1.1]



Priority **P2** · Phase **1**

---

## [1.8] LLM routing: config-driven, all providers routable, record fallbacks, add local Ollama provider

Labels: P2, phase-1, area:llm, type:feature

## Context

`backend/ai/providers/` has 8 provider classes + stub, but `AIGateway._register_providers` (`ai/gateway.py:33`) never registers the Anthropic one, and the routing table in `ai/router.py` only ever routes to `gemini`, `groq`, `nvidia_nim` and `stub`, so the others are never used. When parsing fails, `DecisionAgent` silently falls back to a rule (`suggestions.md` item 4). Companies with sensitive data will want a local model.

## Files

- `backend/ai/router.py`
- `backend/ai/gateway.py`
- `backend/ai/providers/`
- `backend/app/core/config.py`

## What to do

- Load the routing table from config (YAML/env) so any registered provider can be used per task tier.
- Add an Ollama (local) provider.
- Every LLM result records provider, model, tokens, cost and `decision_mode` (`llm` / `fallback` / `rule`); the UI shows fallbacks.
- Log every prompt sent (for the privacy audit in [2.7]).

## Acceptance criteria

- [ ] Tests: routing to each registered provider can be configured (mocked HTTP); a forced parse failure is recorded as `fallback`.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[0.2]



Priority **P2** · Phase **1**

---

## [2.1] DuckDB as the single internal engine (CSV/Parquet become one-table databases)

Labels: P1, phase-2, area:data-connect, area:backend, type:feature

## Context

The relational direction needs one code path for 'a database'. Loading uploads into DuckDB makes CSV a special case of a database, and DuckDB is fast for the aggregations the feature agent will run.

## Files

- `backend/ml/data/ingestion/`
- backend/ml/data/engine.py (new)

## What to do

- A `DataSource` abstraction: `list_tables`, `schema`, `query(sql, params)`, `sample`.
- CSV/Parquet uploads are registered as DuckDB tables in a per-project DuckDB file.
- Existing profiling/leakage/experiments read via the abstraction.

## Acceptance criteria

- [ ] All existing tests and the e2e sample run pass using the DuckDB path.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://duckdb.org

## Depends on

[0.10]



Priority **P1** · Phase **2**

---

## [2.2] Connection manager: Postgres, SQLite, DuckDB connections with safe secret handling

Labels: P1, phase-2, area:data-connect, area:security, area:backend, type:feature

## Context

`POST /ui/data/connect-sql` takes a raw connection string per request and snapshots the result to CSV. Real use needs saved, named connections with secrets kept out of logs and the DB.

## Files

- backend/app/api/v1/connections.py (new)
- `backend/app/db/models/`
- `backend/ml/data/ingestion/sql_loader.py`

## What to do

- CRUD for named connections: type (postgres, sqlite, duckdb; mysql next), host, db, user; password from env var or encrypted at rest (Fernet key from env).
- `POST /connections/{id}/test` checks connectivity and reports the user's privileges.
- Optional SSH tunnel / SSL settings for Postgres (document; implement SSL first).
- Passwords never returned by the API or logged.

## Acceptance criteria

- [ ] Integration test against a Postgres service container in CI: create, test, list (no password in response), delete.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[2.1], [0.9]



Priority **P1** · Phase **2**

---

## [2.3] Read-only enforcement in depth (parser allowlist + read-only transaction + privilege check + limits)

Labels: P1, phase-2, area:security, area:data-connect, type:feature

## Context

Companies will only connect a tool to their database if writes are impossible, not just discouraged. One check is not enough; this needs independent layers, each tested.

## Files

- backend/ml/data/sql_guard.py (new)
- `backend/ml/data/ingestion/sql_loader.py`

## What to do

- Layer 1: parse with `sqlglot` for the target dialect; allow exactly one statement whose tree contains only SELECT (no DML/DDL anywhere, including CTEs; no `SELECT ... INTO`; block functions with side effects such as `pg_sleep`, `lo_import`, `dblink`, `COPY`).
- Layer 2: execute inside a read-only transaction (`SET TRANSACTION READ ONLY` / `default_transaction_read_only=on` for Postgres; `mode=ro` URI for SQLite; `read_only=True` for DuckDB).
- Layer 3: on connect, check the role's privileges; warn loudly in the UI if it can write, and document how to create a read-only role.
- Statement timeout, row limit, and max result size on every query.

## Acceptance criteria

- [ ] A test matrix of at least 15 hostile queries (DML, DDL, data-modifying CTE, multi-statement, `COPY ... TO PROGRAM`, `pg_sleep(1000)`) is refused, against Postgres in CI.
- [ ] With layer 1 disabled in a test, layer 2 still blocks writes (prove the layers are independent).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://github.com/tobymao/sqlglot
- PostgreSQL docs: SET TRANSACTION READ ONLY

## Depends on

[2.2]



Priority **P1** · Phase **2**

---

## [2.4] Schema introspection and relationship graph (declared + inferred keys, time columns)

Labels: P1, phase-2, area:data-connect, type:feature

## Context

The task builder and the feature agent need to know how tables relate and which columns are event times.

## Files

- backend/ml/data/schema_graph.py (new)
- `backend/app/api/v1/connections.py`

## What to do

- Read tables, columns, types, primary keys, declared foreign keys, row counts.
- Infer missing foreign keys by name pattern + value-overlap sampling; mark them `inferred` with a confidence score.
- Detect candidate time columns (date/timestamp types, plus name tokens like `created_at`) per table.
- Expose `GET /connections/{id}/schema` returning a graph (nodes = tables, edges = keys).

## Acceptance criteria

- [ ] On the demo DB ([2.6]), all declared FKs are found, inferred FKs reach >= 90% precision on a version with FKs dropped, and every event table has its time column identified.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelBench schema format: https://github.com/snap-stanford/relbench

## Depends on

[2.2]



Priority **P1** · Phase **2**

---

## [2.5] UI: 'Connect a database' flow with schema graph view

Labels: P1, phase-2, area:frontend, type:feature

## Context

The backend has had `POST /ui/data/connect-sql` since PR #15, but the UI has no way to use it; the landing page only mentions it in text (`frontend/src/Landing.tsx:89`).

## Files

- `frontend/src/Landing.tsx`
- frontend/src/views/ (new Connect view)
- frontend/src/components/SchemaGraph.tsx (new)

## What to do

- Landing and data page offer two equal options: upload a file, or connect a database.
- Form for a named connection (type, host, db, user, password, SSL), 'Test connection' button, privilege warning from [2.3].
- Schema graph view: tables, keys (declared vs inferred styles), row counts, time columns; click a table for columns and stats (no raw rows unless the user allows a preview).

## Acceptance criteria

- [ ] Playwright test against the demo Postgres in CI: connect, see the graph with the expected tables, open a table's details.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[2.4], [0.7]



Priority **P1** · Phase **2**

---

## [2.6] Demo relational database (Postgres in docker-compose) with a realistic schema and planted leaks

Labels: P1, phase-2, area:data-connect, area:ci, type:feature

## Context

Everyone (CI, demos, interviewers, users trying it) needs a realistic multi-table database. A planted leak shows the leakage guarantees working.

## Files

- `docker/docker-compose.yml`
- docker/demo-db/ (new: schema.sql, seed script)

## What to do

- Default: a synthetic e-commerce schema (customers, orders, order_items, products, sessions, support_tickets, refunds), ~10k customers, generated by a seeded script so it's reproducible and has no licence issues.
- Plant: a `customer_status_snapshot` table updated after churn (a future-data trap) and an `is_churned` column on one table.
- Alternative loader for RelBench `rel-f1` into Postgres (used in Phase 6).
- `docker compose up demo-db` gives a ready database plus a read-only role.

## Acceptance criteria

- [ ] `docker compose up demo-db` then `psql` as the read-only user lists the tables; CI uses this service for integration tests.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://github.com/snap-stanford/relbench

## Depends on

[2.2]



Priority **P1** · Phase **2**

---

## [2.7] Privacy controls: what the LLM may see, with an auditable prompt log

Labels: P2, phase-2, area:security, area:llm, type:feature

## Context

Companies' main objection will be 'does our data go to an LLM provider?'. Default must be schema + aggregate stats only, never raw rows.

## Files

- `backend/ai/gateway.py`
- `backend/app/core/config.py`
- `frontend/src/views/Settings.tsx`

## What to do

- Per-project setting: `llm_context = schema_only | schema_and_stats (default) | allow_sample_values`.
- A central context builder enforces the setting; prompts can't be built any other way.
- UI page listing every prompt sent (provider, time, size) with full text on click.
- Document using a local model (Ollama, [1.8]) for 'nothing leaves the network'.

## Acceptance criteria

- [ ] Test: with `schema_and_stats`, no prompt contains any cell value from the demo DB (scan prompts for sampled values).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[1.8], [2.4]



Priority **P2** · Phase **2**

---

## [3.1] Prediction task spec: schema, validation and storage

Labels: P1, phase-3, area:ml, area:leakage, type:feature

## Context

A precise, declarative description of what to predict is what makes point-in-time correctness checkable. Kumo has a 'predictive query language' for this; we define our own open YAML format (idea, not syntax).

## Files

- backend/ml/tasks/spec.py (new)
- docs/task_spec.md (new)

## What to do

- Pydantic model + YAML: `entity_table`, `entity_key`, `entity_filter` (who is eligible at a cutoff), `target` (aggregation over the future window, e.g. `count(orders) = 0`), `horizon` (e.g. 30d), `cutoffs` (start, end, step), `task_type` (binary, regression), `metric`.
- Validation against the schema graph (tables/columns exist, time columns known).
- Store specs per project with versions.

## Acceptance criteria

- [ ] Unit tests: valid specs for churn, late payment and spend-next-30-days on the demo DB parse; invalid ones give clear errors.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelBench task definitions: https://arxiv.org/abs/2407.20060
- Kumo predictive query idea

## Depends on

[2.4]



Priority **P1** · Phase **3**

---

## [3.2] Deterministic training-table builder: labels at cutoff dates

Labels: P1, phase-3, area:ml, area:leakage, area:data-connect, type:feature

## Context

From a task spec, generate one row per (entity, cutoff) with the label computed only from the window (cutoff, cutoff + horizon]. This replaces 'upload a ready-made CSV' for relational data.

## Files

- backend/ml/tasks/labels.py (new)

## What to do

- Compile the spec into SQL (via sqlglot) that produces `entity_id, cutoff_time, label`.
- Eligibility: an entity counts at a cutoff only if it existed before the cutoff (and matches `entity_filter`).
- Run in DuckDB (snapshot) or read-only on the source DB.
- Show the generated SQL and label balance per cutoff in the API response.

## Acceptance criteria

- [ ] Hand-checked test fixtures (tiny DB, 5 customers, 3 cutoffs) produce exactly the expected labels; churn labels on the demo DB match a pandas reference implementation.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Featuretools cutoff times: https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html

## Depends on

[3.1], [2.1]



Priority **P1** · Phase **3**

---

## [3.3] Point-in-time guard: SQL rewriter/validator that forces every feature query to respect the cutoff

Labels: P1, phase-3, area:leakage, area:security, type:feature

## Context

This is MLPilot's core guarantee. Any feature query (human- or LLM-written) must only read rows whose event time is before the row's cutoff. It's enforced by analysing and rewriting the SQL, not by trusting the author.

## Files

- backend/ml/tasks/pit_guard.py (new)

## What to do

- Feature query contract: input is the label table `(entity_id, cutoff_time)`; output is `(entity_id, cutoff_time, value)`.
- Using the sqlglot AST, find every referenced table; for each table with a time column, require (or inject) `t.time_col < cutoff_time` in the join/where scope that touches it, including subqueries and CTEs.
- Reject queries that read time-stamped tables without a provable cutoff filter, reference the label, or use non-deterministic functions (`now()`, `random()`).
- Tables without a time column are allowed only if marked static by the user (e.g. product catalogue), and that marking is shown in the evidence report.

## Acceptance criteria

- [ ] Test suite of at least 25 queries (joins, nested subqueries, CTEs, window functions, aliasing tricks) with expected accept/reject/rewrite results.
- [ ] Property test: for random valid feature queries on the demo DB, deleting all rows after the cutoff doesn't change any feature value.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelAgent enforces 'only source records with timestamp strictly before t': https://arxiv.org/abs/2605.07840
- https://github.com/tobymao/sqlglot

## Depends on

[3.2]



Priority **P1** · Phase **3**

---

## [3.4] Temporal validation for relational tasks (split by cutoff date, not at random)

Labels: P1, phase-3, area:ml, area:leakage, type:feature

## Context

Random splits leak the future when rows are time-stamped. RelBench uses fixed validation and test timestamps.

## Files

- `backend/ml/validation/strategies.py`

## What to do

- Split by cutoff: train on cutoffs < val_time, validate on [val_time, test_time), test on >= test_time; also an expanding-window CV option for feature acceptance ([1.2]).
- Ensure label windows of training rows don't overlap the validation period (gap = horizon).

## Acceptance criteria

- [ ] Test: no training row's label window extends past the first validation cutoff.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelBench temporal splits: https://arxiv.org/abs/2407.20060

## Depends on

[3.2], [1.1]



Priority **P1** · Phase **3**

---

## [3.5] Natural-language question -> task spec, confirmed by the user in chat

Labels: P2, phase-3, area:llm, area:frontend, type:feature

## Context

Users think in questions ('who will churn next month?'). The LLM drafts a spec; deterministic code validates it; the human confirms. AgentDS (2026) found human+AI teams beat AI-only on domain tasks, so the confirmation step is a feature, not friction.

## Files

- backend/ml/tasks/nl_to_spec.py (new)
- frontend/src/components/ (task card)

## What to do

- Prompt with the schema graph (respecting [2.7]) and the question; structured output = task spec.
- Validate ([3.1]); on failure, show the error and ask the LLM once to repair.
- Chat card shows the spec in plain words ('A customer churns if they place no order in the 30 days after the cutoff. Cutoffs: monthly from 2024-01 to 2025-06.'), the generated label SQL, and label balance; buttons: Confirm / Edit.

## Acceptance criteria

- [ ] On 10 example questions for the demo DB (in a test fixture), at least 8 produce a valid spec matching the hand-written expected spec, using the default cheap model; results printed in CI (non-blocking).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- AgentDS: https://arxiv.org/abs/2603.19005

## Depends on

[3.1], [3.2]



Priority **P2** · Phase **3**

---

## [3.6] Leakage canaries: planted future data must be blocked, proven in CI

Labels: P1, phase-3, area:leakage, area:ci, type:feature

## Context

A guarantee is only credible if it's tested against deliberate traps.

## Files

- backend/tests/integration/test_canaries.py (new)
- `docker/demo-db/`

## What to do

- Canary 1: a feature query reading the post-churn `customer_status_snapshot` table without a cutoff -> rejected.
- Canary 2: a 'feature' that is the label shifted by one period -> rejected or flagged by the leakage detector with near-perfect single-feature AUC.
- Canary 3: a static table that's actually updated over time (has `updated_at`) marked static -> warning in the evidence report.

## Acceptance criteria

- [ ] All canaries are blocked/flagged in CI, and the CI log prints a short 'leakage canary report'.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[3.3], [2.6], [1.5]



Priority **P1** · Phase **3**

---

## [4.1] Deterministic baseline features across related tables (DFS-style, cutoff-aware)

Labels: P1, phase-4, area:ml, area:data-connect, type:feature

## Context

Before an LLM proposes anything, MLPilot needs an honest non-LLM baseline: automatic aggregations over related tables (Deep Feature Synthesis). It's also the fallback when no LLM is available, and the bar the LLM must beat.

## Files

- backend/ml/features/dfs.py (new)

## What to do

- For each table reachable from the entity (1-2 hops), generate COUNT/SUM/MEAN/MIN/MAX/days-since-last over windows (7/30/90/365 days before cutoff) for numeric columns, and counts per top categories.
- All generated as SQL passing the point-in-time guard ([3.3]).
- Cap the number of features; drop constant/duplicate ones.
- Option: use Featuretools (BSD-3) directly if it's simpler; compare speed.

## Acceptance criteria

- [ ] On the demo DB churn task, baseline features + LightGBM produce a validation AUROC that's recorded, and every feature passes the guard.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Kanter & Veeramachaneni, Deep Feature Synthesis (DSAA 2015)
- https://github.com/alteryx/featuretools

## Depends on

[3.3], [3.4]



Priority **P1** · Phase **4**

---

## [4.2] LLM SQL feature proposer (one feature at a time, history-aware, structured output)

Labels: P1, phase-4, area:llm, area:ml, type:feature

## Context

The LLM's job is what it's good at: reading table/column meaning and proposing features a human analyst would think of (recency, frequency, trends, ratios across tables). Today's planner proposes arithmetic formulas over one table.

## Files

- `backend/ml/experiments/planner.py`
- backend/ml/features/llm_sql.py (new)

## What to do

- Context: task spec, schema graph (privacy setting from [2.7]), current feature list with gains, rejected features with reasons.
- Structured output: `{name, sql, rationale, expected_effect, tables_used}`.
- Pre-checks before execution: SQL guard ([2.3]), point-in-time guard ([3.3]), non-duplicate (normalised SQL and correlation with existing features after execution).
- If a proposal is rejected by a guard, feed the reason back once for a repaired version.

## Acceptance criteria

- [ ] On the demo DB, 20 proposals with the default cheap model: >= 70% pass the guards, and the pass rate and reasons are printed in a non-blocking CI job.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelAgent: https://arxiv.org/abs/2605.07840
- CAAFE: https://arxiv.org/abs/2305.03403
- LLM-FE: https://arxiv.org/abs/2503.14434

## Depends on

[4.1]



Priority **P1** · Phase **4**

---

## [4.3] Feature execution engine: batch over all cutoffs, cache, time and cost budget

Labels: P2, phase-4, area:data-connect, area:ml, type:feature

## Context

Feature queries run once per (entity, cutoff); doing that naively on a live DB is slow and risky. Execute against a DuckDB snapshot by default.

## Files

- backend/ml/features/executor.py (new)

## What to do

- Snapshot mode (default): copy the needed tables (or a sampled subset of entities) into DuckDB with row limits; live mode optional.
- Execute feature SQL joined to the label table in one pass; cache by (sql hash, data version).
- Per-run budgets: max LLM cost (USD), max wall time, max features; stop cleanly and report when hit.

## Acceptance criteria

- [ ] Test: the same feature is computed once across repeated runs (cache hit); a run with budget $0.05 stops at the budget and says so.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[4.2]



Priority **P2** · Phase **4**

---

## [4.4] Agent loop for relational tasks (replace formula loop; champion path; optional N rollouts)

Labels: P2, phase-4, area:llm, area:ml, type:feature

## Context

Tie it together: baseline -> propose -> guard -> execute -> temporal-CV acceptance ([1.2]) -> champion update ([1.3]) -> repeat until budget or no improvement for K rounds.

## Files

- `backend/ml/agents/decision_agent.py`
- backend/app/api/v1/ (run endpoints)

## What to do

- Run object with status, current champion, history, budget used.
- Stop criteria: budget, max rounds, K rounds without improvement.
- Optional `rollouts=N` runs N independent searches and picks the best on validation (RelAgent uses 5); test set used once at the end.

## Acceptance criteria

- [ ] On the demo DB churn task, a full run completes in CI with the stub provider (deterministic proposals) and the final champion's test metric is reported once.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelAgent multiple rollouts: https://arxiv.org/abs/2605.07840
- AIDE: https://arxiv.org/abs/2502.13138

## Depends on

[4.3], [1.3], [3.4]



Priority **P2** · Phase **4**

---

## [4.5] Chat human-in-the-loop: checkpoints, live narration, settings that actually change the run

Labels: P2, phase-4, area:llm, area:frontend, area:backend, type:feature

## Context

`publish_live_event` and `wait_for_checkpoint` exist in `backend/app/api/v1/chat.py:24,30` but nothing calls them. `POST /chat/settings` parses text into JSON and applies nothing.

## Files

- `backend/app/api/v1/chat.py`
- `backend/ml/agents/decision_agent.py`
- frontend/src/components/ (chat panel)

## What to do

- The run loop publishes narration events (proposal, guard result, CV result, decision) over the existing WebSocket.
- Checkpoints: confirm task spec ([3.5]); optional 'approve each feature' mode; the run waits for a reply with a timeout default.
- Users can veto a feature or suggest one in plain English (goes through the same proposer/guards).
- NL settings ('only tree models', 'optimise recall', 'stop after 20 minutes', 'budget $1') map to real run config fields; the response shows the applied diff.

## Acceptance criteria

- [ ] Integration test: a run in approve mode pauses at a checkpoint, a reply resumes it, a veto removes the feature; a settings message changes the run's budget field.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Master Build Checklist section 5

## Depends on

[4.4]



Priority **P2** · Phase **4**

---

## [5.1] Evidence record per decision and a run report (HTML/Markdown)

Labels: P2, phase-5, area:ml, area:docs, type:feature

## Context

Every decision should read like a security finding: claim, evidence, confidence, what would falsify it. This is what lets a data lead trust and sign off a model built by an agent.

## Files

- `backend/ml/experiments/schema.py`
- backend/ml/reports/ (new)
- frontend/src/views/ (Report view)

## What to do

- For each feature: SQL, rationale, gain with CI and fold scores, guards passed, leakage checks, static-table assumptions.
- For the run: task spec, data version, label balance, splits, champion features, test metrics, LLM cost, prompts log link, what was tried and rejected.
- Export as Markdown and standalone HTML.

## Acceptance criteria

- [ ] Report generated for the demo churn run; every number in it matches the stored records (test compares them).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[4.4]



Priority **P2** · Phase **5**

---

## [5.2] Export bundle: SQL/dbt models + model file + scoring script + task spec, reproducible

Labels: P2, phase-5, area:ml, area:data-connect, type:feature

## Context

No lock-in: the company should be able to run the result without MLPilot. Features as SQL mean they can run inside the warehouse; dbt is how many analytics teams already manage SQL.

## Files

- backend/ml/export/ (new)
- backend/app/api/v1/ui.py (export endpoint)

## What to do

- Bundle: `task.yaml`, `features/*.sql`, a dbt project (`models/features/*.sql` with a `cutoff` var), `model.(txt|joblib)`, `score.py`, `requirements.txt`, `README.md`, and the evidence report.
- `score.py --cutoff YYYY-MM-DD --db URL` computes features and writes predictions to CSV.

## Acceptance criteria

- [ ] CI job: export the demo run, install the bundle in a clean venv, run `score.py` against the demo DB, and reproduce the reported validation metric within 1e-6.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- dbt + ML: https://xebia.com/blog/dbt-machine-learning/

## Depends on

[5.1]



Priority **P2** · Phase **5**

---

## [5.3] Batch scoring: predict for the latest cutoff from the UI/CLI

Labels: P2, phase-5, area:ml, area:backend, type:feature

## Context

The point of the model is a list: 'these 300 customers are most likely to churn this month', with top reasons.

## Files

- backend/app/api/v1/ (predict endpoint)
- `frontend/src/views/`

## What to do

- Endpoint/CLI to score all eligible entities at a given cutoff using the champion; CSV download.
- Top contributing features per row (from [5.4]).
- Writing back to the database is out of scope here (see [7.4]).

## Acceptance criteria

- [ ] Test: scoring at the test cutoff reproduces the stored test predictions.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- suggestions.md item 3

## Depends on

[5.2]



Priority **P2** · Phase **5**

---

## [5.4] Real SHAP explanations, grounded debrief, and grounded Q&A over run history

Labels: P2, phase-5, area:ml, area:llm, type:feature

## Context

After [0.8] the debrief no longer invents SHAP values; this issue adds real ones. The chat's 'why did you reject that feature?' must answer from the stored record, not general knowledge.

## Files

- `backend/app/api/v1/chat.py`
- `backend/ml/reports/`

## What to do

- Add `shap` to requirements; compute TreeSHAP for the champion on a validation sample; store global and per-row values.
- Debrief prompt receives only stored metrics, SHAP summaries and decisions; the response cites the experiment IDs it used.
- Q&A retrieves the relevant experiment records by ID/name before answering; if nothing matches, it says so.

## Acceptance criteria

- [ ] Test: the debrief and Q&A answers for the demo run only mention features and numbers present in the stored records (checked by a string/number match test).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- Master Build Checklist section 5

## Depends on

[4.4]



Priority **P2** · Phase **5**

---

## [5.5] Optional MLflow logging of runs

Labels: P3, phase-5, area:ml, type:feature

## Context

Teams already using MLflow should see MLPilot runs there instead of yet another UI.

## Files

- backend/ml/export/mlflow_logger.py (new)

## What to do

- If `MLFLOW_TRACKING_URI` is set, log params, metrics, features (as SQL artifacts), the model and the report.

## Acceptance criteria

- [ ] Test with a local file-based MLflow store: a run appears with the expected params/metrics/artifacts.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://mlflow.org

## Depends on

[5.1]



Priority **P3** · Phase **5**

---

## [6.1] RelBench harness: run MLPilot on rel-f1 with the official evaluator

Labels: P2, phase-6, area:benchmark, type:research

## Context

Credibility comes from a public, reproducible number. RelBench provides databases, tasks, temporal splits and an evaluator, plus published baselines (LightGBM, RDL/GNN) and newer results (RelAgent, KumoRFM).

## Files

- benchmarks/relbench/ (new)

## What to do

- Load a RelBench dataset into DuckDB (or the demo Postgres), map its tasks to MLPilot task specs.
- Run baseline features, DFS, and MLPilot agent; evaluate test predictions with RelBench's evaluator.
- Start with `rel-f1` (small); then `rel-trial` and one more that fits a laptop.
- Record LLM model, cost, tokens and wall time.

## Acceptance criteria

- [ ] `make benchmark DATASET=rel-f1` reproduces the numbers on a second machine within normal seed variance (report mean +- std over 3 seeds).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelBench: https://arxiv.org/abs/2407.20060 and https://github.com/snap-stanford/relbench
- RelBench v2: https://arxiv.org/abs/2602.12606

## Depends on

[4.4]



Priority **P2** · Phase **6**

---

## [6.2] Publish the benchmark table (with losses) in the README

Labels: P2, phase-6, area:benchmark, area:docs, type:docs

## Context

Compare honestly: LightGBM on the entity table only, DFS + LightGBM, MLPilot with a cheap LLM, MLPilot with a strong LLM, and published numbers for RDL, RelAgent and KumoRFM (cited, not re-run). Include cost and time per task.

## Files

- `README.md`
- benchmarks/RESULTS.md (new)

## What to do

- Table + method notes + exact commands; mark which numbers are ours vs quoted; include tasks where MLPilot loses.

## Acceptance criteria

- [ ] Every 'ours' number links to a reproducible command; every quoted number links to its paper/table.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- RelAgent: https://arxiv.org/abs/2605.07840
- KumoRFM-2: https://arxiv.org/abs/2604.12596

## Depends on

[6.1]



Priority **P2** · Phase **6**

---

## [6.3] Single-table sanity benchmark vs AutoGluon on OpenML (backup-floor check)

Labels: P3, phase-6, area:benchmark, type:research

## Context

For CSV users, show where MLPilot stands against the strongest free tool. Expect AutoGluon to win on accuracy; the point is honesty and cost/time.

## Files

- benchmarks/openml/ (new)

## What to do

- 8-10 OpenML classification datasets; baseline vs MLPilot vs AutoGluon (time-limited); report AUROC, time, LLM cost.

## Acceptance criteria

- [ ] Reproducible script and a results table.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://auto.gluon.ai
- https://www.openml.org

## Depends on

[1.7]



Priority **P3** · Phase **6**

---

## [7.1] Package: `pip install mlpilot` CLI and one-command Docker

Labels: P2, phase-7, area:ci, type:feature

## Context

Adoption needs a 5-minute path: install, point at a database, run.

## Files

- `backend/pyproject.toml`
- mlpilot/cli.py (new)
- `docker/`

## What to do

- CLI: `mlpilot connect`, `mlpilot schema`, `mlpilot task create --question ...`, `mlpilot run`, `mlpilot report`, `mlpilot export`, `mlpilot ui`.
- Docker image running API + UI; `docker run -p 8000:8000 ... mlpilot`.
- Release workflow publishing to PyPI/GHCR on tags.

## Acceptance criteria

- [ ] On a clean machine: `pip install mlpilot` then the README's 5 commands produce a report on the demo DB.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[5.2]



Priority **P2** · Phase **7**

---

## [7.2] MCP server so AI assistants can use MLPilot as a tool

Labels: P3, phase-7, area:llm, type:feature

## Context

Teams increasingly work through AI assistants (Claude, Cursor). Exposing MLPilot's guarded operations as MCP tools lets an assistant ask for a prediction task on a database while MLPilot keeps the guarantees.

## Files

- mlpilot/mcp_server.py (new)

## What to do

- Tools: `describe_schema`, `draft_task`, `run_task` (budgeted), `get_report`, `score`. All go through the same guards; no raw-SQL execution tool.

## Acceptance criteria

- [ ] An MCP client test lists the tools and runs `draft_task` + `run_task` on the demo DB with the stub provider.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## References

- https://modelcontextprotocol.io

## Depends on

[7.1]



Priority **P3** · Phase **7**

---

## [7.3] Docs site, 2-minute demo video and hosted demo

Labels: P3, phase-7, area:docs, type:docs

## Context

People decide in two minutes. A hosted demo (e.g. Hugging Face Spaces) on the demo DB with the stub or a cheap model shows the whole flow without setup.

## Files

- `docs/`
- `README.md`

## What to do

- Docs: quick start, task spec reference, safety model (read-only layers, point-in-time guard, privacy), benchmark. Demo video. Hosted demo with spend cap.

## Acceptance criteria

- [ ] Someone outside the project follows the docs to run MLPilot on their own Postgres and reports back (link the feedback).

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[7.1], [6.2]



Priority **P3** · Phase **7**

---

## [7.4] Later: write predictions back, scheduled re-scoring, more connectors

Labels: P3, phase-7, area:data-connect, type:feature

## Context

Only after Phase 6. These turn a one-off model into an ongoing service, but each needs write access or scheduling, so they come last and are opt-in.

## What to do

- Opt-in write-back to a dedicated schema (`mlpilot_predictions`) using a separate, explicitly configured write connection.
- Scheduled re-scoring and a simple comparison of feature distributions between runs.
- Connectors: MySQL, Snowflake, BigQuery.

## Acceptance criteria

- [ ] Split into separate issues before starting; each with its own tests.

Done means every box above is shown passing in CI or pasted command output, not self-reported.

## Depends on

[6.2]



Priority **P3** · Phase **7**

---
