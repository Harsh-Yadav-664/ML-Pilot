# AGENTS.md: rules for anyone (human or AI agent) working on MLPilot

Read this file fully before changing anything. It applies to Claude Code, Antigravity, Copilot, Cursor and humans alike.

## 1. What MLPilot is

MLPilot is an open-source, self-hosted **prediction agent that works directly on a company's relational database**.

A user connects read-only to a database (Postgres, SQLite, DuckDB; CSV/Parquet are treated as a one-table database), asks a prediction question ("which customers will stop ordering in the next 30 days?"), and MLPilot:

1. drafts a **prediction task spec** (entity, target, horizon, cutoff dates) that the user confirms;
2. builds a **point-in-time-correct training table** (each row only uses data from before its cutoff date);
3. lets an LLM propose features **as readable SQL**, one at a time, and keeps a feature only if time-based validation shows a real gain;
4. outputs an **evidence report** and an **export bundle** (SQL/dbt features, model file, scoring script).

Full reasoning, competitors, papers and phases: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Work items: GitHub issues titled `[phase.n] ...`, listed in tracking issue [#21](https://github.com/Harsh-Yadav-664/ML-Pilot/issues/21). GitHub is the only source for issues: edit them there. v1 is done when issue #102 ([5.6], the end-to-end acceptance test) is green on main.

## 2. Non-negotiable rules

These rules exist because breaking any one of them makes the product untrustworthy. If a task seems to need breaking one, stop and say so in the PR instead.

1. **The LLM proposes; deterministic code validates and executes.** LLM output is data, never code. Never `eval`/`exec` LLM output, never run LLM-written Python or shell. Formulas go through the whitelist AST evaluator; SQL goes through the SQL guard and the point-in-time guard.
2. **Read-only on user databases.** Every query against a user database goes through the SQL guard (single `SELECT` only, parsed with `sqlglot`), runs in a read-only transaction, with a statement timeout and row limit. No code path may write to a user database unless an issue explicitly says so (Phase 7+, opt-in, separate connection).
3. **No future data in training.** Relational features must pass the point-in-time guard (only rows with event time before the row's cutoff). Relational tasks use time-based splits, never random splits.
4. **Honest statistics.** Never tune or select on the test set. The test set is scored once per run, at the end. A feature or decision is accepted by code using validated gains, not by the LLM.
5. **No fake numbers.** No `Math.random()`, simulated metrics, hardcoded SHAP values or mock replies outside an explicit, visibly labelled Demo mode. If something isn't implemented, the UI says "not available", not a made-up value.
6. **Privacy by default.** By default the LLM sees schema and aggregate column stats only, never raw cell values. All prompts are built through the central context builder and logged.
7. **Secrets never leak.** Never log, return or commit passwords, API keys or full connection strings. `.env` files stay untracked.
8. **Failures are loud.** Don't swallow exceptions and return empty results. Record `failed` with a message, or return an HTTP error. LLM fallbacks are recorded as `decision_mode: fallback` and shown in the UI.
9. **Licences:** only add dependencies with permissive licences (MIT, BSD, Apache-2.0, PostgreSQL, etc.). Do not add non-commercial or "not for production" packages (e.g. RealTabPFN-2.5 weights, getML community) as defaults.
10. **Don't claim what isn't true.** README, UI copy and PR descriptions only describe behaviour that a test or CI step demonstrates.

## 3. How to pick up and finish an issue

1. Pick the **lowest-numbered open issue in the earliest unfinished phase** whose `Depends on` issues are closed. Phase 0 comes before everything.
2. Read the whole issue: Context, Files, What to do, Acceptance criteria. Open every file it lists before writing code; line numbers may have shifted, so search for the named function or string.
3. Stay inside the issue's scope. If you find another bug, note it in the PR description (or open an issue); don't fix it in the same PR unless it blocks the acceptance criteria.
4. Write or update tests first where possible. Every acceptance criterion needs a test, a CI step, or pasted command output as proof.
5. Run the checks in section 5 locally. All must pass.
6. Open **one PR per issue**, titled like the issue, with `Fixes #N` and the PR template filled in, including the **proof** for each acceptance criterion.
7. **Done means proven.** Never tick a box or write "done" because the code was written. A previous round of work was reported complete and was not; reviewers will check.

Branch names: `issue-<number>-<short-slug>` unless your tool assigns one.

## 4. Repository map

| Path | What it is |
|---|---|
| `backend/app/` | FastAPI app. `api/v1/ui.py` serves the UI (`/api/v1/ui/...`); `api/v1/chat.py` the chat panel |
| `backend/ai/` | LLM gateway: providers, routing (`router.py`), cost tracking. Use `AIGateway.complete` / `complete_structured` |
| `backend/ml/data/` | Ingestion (CSV, Parquet, SQL), profiling, preparation |
| `backend/ml/validation/` | Leakage detection, validation strategies |
| `backend/ml/experiments/` | Planner, executor (safe AST evaluator, training), runner |
| `backend/ml/agents/` | Decision agent (experiment loop), cleaning agent |
| `backend/tests/` | `unit/` and `integration/` tests (pytest, asyncio auto mode) |
| `frontend/` | React + Vite UI started by `start.py`. Sample data only in the explicit Demo mode |
| `docker/` | docker-compose (demo Postgres DB planned in [2.6]) |
| `docs/` | Architecture notes, `ROADMAP.md` |

New modules planned by the roadmap (create them where the issues say): `backend/ml/data/sql_guard.py`, `backend/ml/data/schema_graph.py`, `backend/ml/tasks/` (task spec, labels, point-in-time guard), `backend/ml/features/` (safe eval, DFS, LLM SQL features), `backend/ml/reports/`, `backend/ml/export/`, `benchmarks/`.

## 5. Commands

```bash
# Backend (Python version from backend/pyproject.toml)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt  # app deps + pinned ruff and mypy
uvicorn app.main:app --reload        # API on http://localhost:8000
pytest                               # all backend tests
ruff check .                         # lint (CI fails on any error)
ruff format --check .                # formatting (run `ruff format .` to fix)
mypy                                 # type check of ai/, ml/, app/

# Metadata DB schema changes (Alembic owns the schema; never create tables from the models directly)
alembic revision --autogenerate -m "what changed"   # after editing app/db/models/, then review the file
alembic upgrade head                 # the app also runs this on start-up

# Frontend
cd frontend
npm ci
npm run build                        # must pass
npm run dev                          # UI on http://localhost:5173

# Both together
python start.py
```

Without API keys the gateway uses the offline `stub` provider. Tests must pass with the stub provider and no network.

Note: on `main` at 2026-10-05 the install and start-up are broken (issues [0.1] and [0.2]). Fix those first.

## 6. Coding conventions

- Python: type hints, pydantic models for API and LLM structured output, `async` FastAPI handlers; run CPU-heavy work with `asyncio.to_thread`.
- One shared implementation per concern: one safe evaluator, one SQL guard, one point-in-time guard, one LLM context builder. Don't copy them into endpoints.
- Every experiment/run record stores what's needed to reproduce it: data version hash, task spec, split definition, seed, engine and version, features (formula or SQL), LLM provider/model/cost, `decision_mode`.
- LLM prompts live next to the code that uses them, ask for structured output, and are covered by a stub-provider test.
- Keep the stub provider deterministic so tests are repeatable.
- Frontend: all API calls go through `frontend/src/api/`; base URL from `VITE_API_BASE_URL`.

## 7. Scope guard (do not build)

Unless an issue explicitly asks for it: auto-joining external data sources, drift healing, one-click deployment, forecasting curves, NLP or computer-vision models, writing to user databases, new LLM providers, new UI pages with no backend behind them. Later ideas go in a new issue, not in your PR.

## 8. When unsure

Prefer the safer, simpler choice, state the assumption in the PR description, and continue. If the safe choice would break a rule in section 2, stop and ask in the issue.
