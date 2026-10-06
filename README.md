# MLPilot

An open-source, self-hosted prediction agent for tabular and relational data. **Early stage:** this README separates what runs today from what is planned.

## The problem

Most of a company's prediction questions ("which customers will stop ordering next month?") are answered from data spread across several database tables. Before any model can be trained, someone has to turn those tables into one training table with the right time cutoffs. Get a cutoff wrong and future information leaks into training: the model looks great in testing and fails in production. Free AutoML tools and LLM data-science agents all start from a ready-made CSV, so they skip this step.

## The solution (where MLPilot is going)

MLPilot connects read-only to a company database. It turns a question into a reviewable prediction task, builds a point-in-time-correct training table, and lets an LLM propose features as readable SQL. It keeps a feature only if time-based validation shows a real gain. It ends with an evidence report and exportable SQL and model files. The LLM only proposes; deterministic code validates and runs everything.

The reasoning, competitors and phases are in [`docs/ROADMAP.md`](docs/ROADMAP.md). Work is tracked in GitHub issues, listed in [issue #21](https://github.com/Harsh-Yadav-664/ML-Pilot/issues/21).

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
- API: http://localhost:8000 (interactive docs at http://localhost:8000/docs)
- API keys are optional. Without them, MLPilot uses the offline stub provider, whose suggestions are fixed placeholders. To use a real LLM, copy `backend/.env.example` to `backend/.env` and fill in a key. Never commit `.env`.

On Windows PowerShell, set `$env:PYTHONIOENCODING="utf-8"` before `python start.py`.

### Run the checks

```bash
cd backend && pip install -r requirements-dev.txt && pytest && ruff check . && ruff format --check . && mypy
cd frontend && npm run build && npx tsc --noEmit
cd frontend && npx playwright install chromium && npm run test:smoke   # starts backend + UI itself
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
