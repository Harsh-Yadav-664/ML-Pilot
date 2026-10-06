# MLPilot

**Ask your database a prediction question. Get an answer you can check.**

> "Which customers who ordered this year won't order again in the next 30 days?"

MLPilot is a self-hosted tool for questions like that. You point it at your data, it builds the training table with the dates handled correctly, tries ideas for new columns one at a time, and keeps only the ones that survive an honest test. At the end you get the model, the SQL that builds its inputs, and a record of what was tried and what actually helped.

**Status: early.** Today it works on a single CSV file. Connecting to a database from the UI, joining several tables and the final report are not built yet. [What works today](#what-works-today) lists what a test in CI shows, and [what doesn't](#not-working-yet-or-planned) lists the rest. Nothing in this README is claimed unless a test or CI step demonstrates it.

## Why this exists

The model is the easy part: LightGBM is three lines. Teams lose weeks on what comes before it. What does "churn" mean on a given date? How do five tables become one row per customer? And does any column quietly contain information from after that date?

That last one is called leakage, and it is easy to do by accident. A column like "total orders" counts next month's orders too, the model scores 95% in testing, and in production it does nothing. Free AutoML tools and chat-style data assistants mostly start from a clean CSV, so they start after this step. MLPilot is meant to start before it.

## What you can rely on

These come from [`AGENTS.md`](AGENTS.md), where they are rules for every contributor, human or AI:

- **The AI suggests, code decides.** A language model may propose a feature, but what it writes is treated as data. Formulas go through a whitelist evaluator, it never runs Python or shell from a model, and whether a feature is kept is decided by a statistical rule, not by the model.
- **The test set is used once.** Tuning and every keep or reject decision use training and validation rows only. A test checks that the test rows are scored exactly once, after all tuning.
- **No made-up numbers.** If something isn't implemented, the screen says "not available". When an LLM call falls back to the offline stub, the result is marked as a fallback.
- **Read-only by design.** It runs on your machine, the SQL import accepts a single read-only `SELECT` and refuses writes, and a database password is kept out of error messages.

## Who it is for

This is the target, not today's reach, because today it reads one CSV:

- a small data team or analyst with a Postgres or SQLite database and no ML engineer, who wants a first churn or repeat-purchase model they can explain to a manager;
- a consultant who needs to show a client evidence, not just a score;
- anyone who wants to see the whole pipeline done carefully, with the code to read.

The reasoning, the competitors and the phases are in [`docs/ROADMAP.md`](docs/ROADMAP.md). Work is tracked in GitHub issues, listed in [issue #21](https://github.com/Harsh-Yadav-664/ML-Pilot/issues/21). Each issue says what to build, which files to open and how to prove it is done.

## What works today

Today MLPilot works on a **single CSV file**. Each item below is exercised by a test that runs in CI ([workflow](.github/workflows/ci.yml)). All tests use the offline stub LLM provider, so they need no API keys and no network.

| Works today | Proven by |
|---|---|
| Load the bundled telecom churn sample, or upload your own CSV | `test_api_security.py::test_sample_dataset_is_allowed`, `test_ui_data_api.py::test_upload_csv_returns_columns_and_rows` |
| Per-column profile (type, role, missing %, distribution) | `test_ui_data_api.py::test_columns_profile_sample` |
| Leakage scan: each finding has a taxonomy category (L1–L4, ID), a severity and its evidence. On the telecom sample only `customerID` is flagged; on a 13-dataset planted-leak suite (10 planted leaks, 3 clean controls) it finds 10/10 with 0 false flags. The suite was built alongside the checks, so treat these numbers as a regression test, not an independent benchmark | `test_leakage.py`, CI step `Planted-leak suite`, `test_ui_data_api.py::test_leakage_warnings_carry_category` |
| Baseline model runs to `completed`, or is recorded as `failed` with a message | `test_baseline_lifecycle.py` |
| String labels such as `Yes`/`No` are encoded for training and decoded back in the exported script | `test_targets.py`, `test_agent_loop_telecom.py` |
| Agent loop: baseline plus 2 LLM-proposed experiments on the telecom sample, all reaching `completed` with numeric metrics | `test_agent_loop_telecom.py` |
| Experiment suggestions that take earlier proposals into account | `test_planner.py`, `test_suggestions_api.py` |
| Export a standalone training script for an experiment | `test_agent_loop_telecom.py` |
| Chat panel answers, and a debrief that uses the model's built-in feature importances (not SHAP) | `test_chat_and_autoclean_api.py`, `test_debrief.py` |
| Auto-clean returns an HTTP error when the LLM's cleaning plan is invalid (no silent empty result) | `test_cleaning_agent.py`, `test_chat_and_autoclean_api.py::test_auto_clean_surfaces_llm_failure` |
| LLM gateway falls back to the offline stub when a provider fails or no key is set | `test_ai_gateway.py`, `test_stub_provider.py` |
| The API takes no file paths: data is addressed by content-hash version id, and other projects' experiments return 404 | `test_api_security.py` |
| SQL import (API only) accepts a single read-only `SELECT`, refuses writes, and hides passwords in errors | `test_sql_loader.py`, `test_api_security.py` |
| The web UI builds and type-checks | CI `frontend` job (`npm run build`, `tsc --noEmit`) |
| In a real browser, the UI loads the sample through the project API and the baseline run completes | CI `ui-smoke` job (`npm run test:smoke`, Playwright) |
| The frontend's TypeScript API types match the backend's response models | CI: `python -m scripts.export_openapi --check` and `npm run gen:api` + `git diff --exit-code` |
| Training and the agent loop run as durable jobs: an interrupted job is reported `failed: interrupted by restart`, a cancelled one stops before its next step, and progress is an ordered event log (poll or SSE) | `test_job_runner.py`, `test_agent_loop_telecom.py` |
| Every API call needs the local access token (401 without it); only the UI origins pass CORS; `python start.py` sets the token up with no manual step | `test_local_token_and_cors.py`, CI `ui-smoke` |
| Binary tasks report base rate, PR-AUC, precision/recall/lift at the top 1/5/10% and top 100, Brier and ECE for validation and test; the probability threshold and any calibration are fitted on validation only | `test_business_metrics.py`, `test_threshold_and_calibration_split.py`, `test_agent_loop_telecom.py` |
| A CSV or Parquet file becomes a table in the project's own DuckDB file, and profiling, the leakage scan and training all read it from there; the file can't read or write anything outside its own tables, and loading matches `pandas.read_csv` types, so `Yes`/`No` targets stay text. A 1M-row, 10-column CSV loads and profiles in under 10 s | `test_duckdb_engine.py` (the test prints its timing) |
| Save named connections to a Postgres, SQLite or DuckDB database; the password is read from an environment variable or stored encrypted (and refused without a key), is never returned or logged (a DEBUG-level connect, test and query cycle is scanned for it), and every query is one `SELECT` in a read-only session. The test endpoint reports the server version and whether the role could write | `test_connections_api.py` (runs against a Postgres service in CI), `test_live_sources.py`, `test_secrets_and_redaction.py` |
| Every completed experiment stores a run manifest (data version, split, seed, engine and parameters, features, LLM calls, metrics) that contains no secrets, and replaying it without the LLM reproduces the validation and test metrics to 1e-9 | `test_rerun.py`; format in `docs/experiment_schema.md` |

### Not working yet, or planned

- **Direct database connection in the UI: planned (Phase 2).** The backend can import one read-only SQL query into a CSV (`POST /api/v1/projects/{project_id}/datasets/sql`), but the UI does not offer it, and there is no multi-table support yet.
- **Prediction task spec, point-in-time training tables, LLM-written SQL features, evidence report, SQL/dbt export:** planned (Phases 1–5).
- **Real LLM providers:** the gateway registers a real provider when you set its key (see `backend/.env.example`), but CI never calls a real provider, so that path is untested.
- **Demo mode** in the UI shows sample charts and numbers. It is switched on explicitly and labelled on screen. Everything outside Demo mode comes from the backend or says "not available".

## Quick start

You need Python 3.13 and Node.js 20.19 or newer (Vite 7 requires it). From the repository root:

```bash
# 1. Backend dependencies
python3.13 -m venv backend/.venv
source backend/.venv/bin/activate          # Windows: backend\.venv\Scripts\activate
pip install -r backend/requirements.txt

# 2. Frontend dependencies
cd frontend && npm ci && cd ..

# 3. Start both servers (run with the venv's Python)
python start.py
```

- UI: http://localhost:5173 (click "Start with sample data" to try the telecom churn data)
- API: http://127.0.0.1:8000 (interactive docs at http://127.0.0.1:8000/docs). The servers listen on this machine only.
- Every API call needs a local access token. `start.py` creates it on first run in `~/.mlpilot/token` (owner-only) and passes it to the UI through the git-ignored `frontend/.env.local`, so there is nothing to set up. Scripts send `Authorization: Bearer $(cat ~/.mlpilot/token)`. Only the UI's origins may call the API from a browser (`MLPILOT_CORS_ORIGINS`, default `http://localhost:5173`). This is not multi-user auth. To listen on other interfaces, set `MLPILOT_HOST`, and MLPilot logs a warning unless `MLPILOT_TOKEN` is also set.
- API keys are optional. Without them, MLPilot uses the offline stub provider, whose suggestions are fixed placeholders. To use a real LLM, copy `backend/.env.example` to `backend/.env` and fill in a key. Never commit `.env`.

On Windows PowerShell, set `$env:PYTHONIOENCODING="utf-8"` before `python start.py`.

### Run the checks

```bash
cd backend && pip install -r requirements-dev.txt && pytest && ruff check . && ruff format --check . && mypy
cd frontend && npm run build && npx tsc --noEmit
cd frontend && npx playwright install chromium && npm run test:smoke   # runs python start.py itself
```

## Architecture

```
React + Vite UI (frontend/)  --HTTP-->  FastAPI (backend/app/)
                                             |
                    +------------------------+------------------------+
                    |                                                 |
       LLM gateway (backend/ai/)                         ML engine (backend/ml/)
       providers + routing + cost tracking,              ingestion, profiling, leakage scan,
       offline stub fallback                             planner, safe executor, decision agent
                                                                      |
                                                       SQLite run store (experiments, results)
```

- **`backend/ai/`**: the LLM gateway. Every LLM call goes through `AIGateway.complete` or `complete_structured`. LLM output is treated as data and validated, never executed as code.
- **`backend/ml/`**: data loading (CSV, read-only SQL), profiling, the leakage detector, the experiment planner and the executor. Feature formulas run through a whitelisted AST evaluator.
- **`frontend/`**: the UI. All API calls live in `frontend/src/api/`, and the base URL comes from `VITE_API_BASE_URL`.

Contributor and agent rules are in [`AGENTS.md`](AGENTS.md).
