# MLPilot roadmap and product direction

This is the product direction and phased plan behind the GitHub issues listed in [#21](https://github.com/Harsh-Yadav-664/ML-Pilot/issues/21). Research done 2026-10-05 against `main` at `77a3d81`. Items marked **(verified)** were checked in code or a cited source; **(inferred)** marks judgement.

---

## 0. The answer in one page

**Stop building "AutoML, but cheaper" for a single CSV.** That space is full of free, strong tools (AutoGluon, ChatGPT/Julius, Colab's and Databricks' data-science agents), and a small open-source project cannot beat them at fitting models.

**Build this instead:** *MLPilot becomes the open-source, self-hosted "prediction agent" that works directly on a company's own database.*

You connect it read-only to Postgres/MySQL/SQLite (later Snowflake/BigQuery), ask a question such as *"which customers will stop ordering in the next 30 days?"*, and MLPilot:

1. turns the question into a precise, reviewable **prediction task** (who, what, when, how far ahead);
2. builds a **point-in-time-correct training table** from many tables, so no future information leaks into training (enforced by code, not by asking the LLM nicely);
3. has an LLM propose features **as readable SQL** over related tables (counts, recency, averages, trends), runs each one safely, and keeps it only if honest time-based validation shows it helps;
4. hands back an **evidence report** (each feature's SQL, why it was proposed, how much it helped, which leakage checks it passed) plus **exportable SQL/dbt models and a trained model** the company owns and can run without MLPilot.

**Why this is the right wedge (short version):**

- **It's the hardest, most error-prone step in real company ML, and free tools skip it.** Every free AutoML and every Kaggle-style agent (AIDE, MLE-STAR, AutoKaggle, AutoGluon Assistant) starts from a ready-made `train.csv`. In a company, the data lives in 10–50 related tables, and someone has to build that CSV by hand with correct time cutoffs. Getting the cutoff wrong is the classic leakage bug that produces models that look great and fail in production.
- **The market has just priced it.** Kumo.ai built exactly this capability (predictions directly over relational data), is used by DoorDash, Reddit and Sainsbury's, and was **acquired by NVIDIA in June 2026 (reported at over $400M)**. It's proprietary and API-only. Its documented connectors don't list Postgres. (verified, sources in §12)
- **Research shows the approach works, but nobody has made it usable.** *RelAgent* (NeurIPS 2026) showed an LLM agent writing SQL feature queries plus a classical model is competitive with the best relational models on the RelBench benchmark, and best on RelBench v2. It's research code that runs on benchmark files in DuckDB, not on a live company database. (verified)
- **It can be proven for free.** RelBench is a public benchmark with published numbers for LightGBM, graph neural networks, KumoRFM and RelAgent. We can publish an honest table: *"open-source, runs on a free/cheap LLM, X% of KumoRFM's accuracy at ~$Y per task."* The comparison becomes one supporting fact, not the whole pitch.

**What stays from today's code:** the LLM gateway, the safe evaluator idea, experiment records, script export, the leakage detector (rewritten), the chat panel and the UI shell. **What goes:** mock data, fake SHAP, the duplicate frontend, the "Deployments/drift" mock screens, and the claim that MLPilot is a better general AutoML.

**First, though, the basics.** Phase 0 makes `main` install, start and run end to end from a fresh clone, with no fake numbers in the UI. Nothing else matters until that's true.

---

## 1. MLPilot explained

### 1.1 The problem, in plain words

A company wants to predict something: which customers will churn, which invoices will be paid late, which leads will convert. The data already exists in their database. Getting from "data in tables" to "a model we can trust" takes weeks, and most of that time isn't spent on the model. It goes into:

- deciding exactly what to predict and for which date ("churn" means *no order in the 30 days after 1 March*, not "ever");
- joining many tables into one row per customer, using **only data available before that date**;
- inventing useful features (orders in the last 90 days, days since last login, average basket size, trend…);
- checking that nothing "from the future" sneaked in (leakage);
- proving to their manager that the model is real and not a fluke;
- putting the feature logic somewhere it can run again next month.

The model itself (LightGBM/XGBoost) takes 3 lines. Everything else is where data scientists spend their time, and where mistakes happen.

### 1.2 What MLPilot does about it (the target product)

| Step | What MLPilot does | Who decides |
|---|---|---|
| Connect | Read-only connection to the database; reads schema, keys and simple stats | Code |
| Ask | User types the question; LLM drafts a **task spec** (entity, target, horizon, cutoff dates) | LLM drafts, **human confirms** |
| Build labels | Deterministic SQL generates one row per (entity, cutoff date) with the true outcome | Code |
| Baseline | Simple automatic features + LightGBM, time-based validation | Code |
| Propose features | LLM proposes one SQL feature at a time, using history of what helped | LLM proposes |
| Check & run | SQL is parsed, whitelisted, forced to respect the cutoff, executed read-only | Code |
| Accept / reject | Keep only if validated improvement beats noise | **Code (statistics), not the LLM** |
| Review | Human can veto or add features in chat; every decision is explained | Human |
| Deliver | Evidence report + SQL/dbt export + model file + scoring script | Code |

The project's core rule: **the LLM proposes, deterministic code validates and executes.**

### 1.3 How the code works today (map)

| Part | Files | What it is | State |
|---|---|---|---|
| API | `backend/app/main.py`, `backend/app/api/v1/*.py` | FastAPI routes; the UI uses `/api/v1/ui/...` in `ui.py` | Won't start on a fresh clone (bad import) |
| LLM gateway | `backend/ai/gateway.py`, `ai/router.py`, `ai/providers/*` | One interface over 8 LLM providers + offline stub, with fallback and a cost tracker | Good. Only Gemini, Groq, NVIDIA NIM (+stub) are routed to; Anthropic is never even registered |
| Data in | `backend/ml/data/ingestion/csv_loader.py`, `parquet_loader.py`, `sql_loader.py` | Load CSV/Parquet; SQL loader runs a query and snapshots to CSV | SQL path exists in the backend (`POST /ui/data/connect-sql`) but **there is no UI for it**, and its "read-only" check is a `startswith("select")` |
| Profiling | `backend/ml/data/profiling/*` | Column stats | OK |
| Cleaning | `backend/ml/agents/cleaning_agent.py`, `ml/data/preparation/*` | LLM suggests a cleaning recipe | Calls `gateway.chat_completion`, which doesn't exist, so it silently uses an empty recipe |
| Leakage | `backend/ml/validation/leakage.py` | 7 `detect_*` checks (docs say 6) | Substring matching produces false positives; correlation check skips categorical targets |
| Experiments | `backend/ml/experiments/executor.py`, `planner.py`, `runner.py` | Applies a feature formula (safe AST), trains, tunes with Optuna, records metrics | Tunes **and** reports on the test set; invalid formulas become a column of zeros |
| Agent | `backend/ml/agents/decision_agent.py` | Loop: propose one feature → run → keep/reject | Shape is right; keep/reject is an LLM comparing two numbers; features never accumulate |
| Chat | `backend/app/api/v1/chat.py` | Steering, Q&A, debrief, checkpoints | Calls `gateway.route_request` (doesn't exist) and imports a missing module; SHAP values are hardcoded (`age`, `balance`…); `shap` isn't installed; checkpoints/narration never called |
| Frontend | `frontend/src`, `frontend/new_ui/src` | React UI (two near-identical copies) | Calls wrong URL prefix, falls back to mock data silently; upload stats are `Math.random()`, metrics are simulated |
| Tests | `backend/tests/unit/*` | 47 unit tests | 1 fails (calls removed method); no CI |

### 1.4 Key terms

- **Point-in-time correctness:** when building a training row "as of 1 March", every feature may only use rows with a timestamp before 1 March. Breaking this is *temporal leakage*.
- **Entity / cutoff / horizon:** entity = the thing predicted (a customer); cutoff = the "as of" date; horizon = how far ahead (30 days).
- **Temporal validation:** train on earlier cutoffs, validate on later ones. A random 80/20 split leaks the future.
- **Nested tuning:** tune hyperparameters on validation data, report once on untouched test data. Today MLPilot tunes on the test set.
- **Relational deep learning (RDL):** treat the database as a graph and train a graph neural network on it (Stanford's RelBench work). Strong but heavy.
- **Relational foundation model:** a model pre-trained on many databases that predicts without training (KumoRFM). Strong, proprietary.
- **SQL feature program:** RelAgent's idea: features are SQL queries, so they're readable and can run inside the database in production.

---

## 2. Why not a one-stop ML platform

A broad "one-stop ML/AI platform" is the obvious alternative. We chose a narrow wedge instead, for these reasons:

- **Breadth is where the incumbents are strongest.** DataRobot, H2O, Vertex AI, SageMaker, Databricks and AutoGluon have teams of tens to hundreds of people and years of head start. A one-stop platform from a small project will be shallower than each of them at every step. Users try it, compare it with AutoGluon, and leave.
- **Free tools already cover "upload CSV, get a model".** AutoGluon (open source, top of the TabArena benchmark), PyCaret, FLAML; ChatGPT data analysis and Julius for non-coders; Colab's Data Science Agent, BigQuery's and Databricks' data-science agents inside their platforms.
- **What wins adoption is one painful job done much better.** That's how Optuna, MLflow and dbt each became standard while big platforms already "had" those features.
- **The relational wedge still feels like a "complete" product to the user**, because it covers one job end to end: from their database to a trustworthy model and the SQL to run it. That gives users the "one-stop" experience for a real job without trying to cover all of ML.
- **It can grow outward later.** Once the core works: write predictions back to the database, scheduled re-scoring, monitoring, more connectors, and an MCP server so other AI agents can call it. Each extension stays tied to the same core.

---

## 3. The landscape: who does what, and what we can adopt legally

### 3.1 Open-source tabular AutoML

| Tool | What it's good at | Relational DB / time cutoffs? | Adopt (legally) |
|---|---|---|---|
| **AutoGluon** (Apache-2.0) | Best open-source accuracy on single tables; stacking; tabular foundation models built in | No; needs one table | Use it **as an optional model engine** (`pip install`); don't compete with it |
| **AutoGluon Assistant / MLZero** (Apache-2.0) | Multi-agent, zero-code ML from a folder of files | No point-in-time handling | Idea: agent reads data description + files; compare in benchmark |
| PyCaret, FLAML, H2O AutoML | Fast baselines, many models | No | Not needed; LightGBM + optional AutoGluon covers it |
| **Featuretools** (BSD-3) | Deep Feature Synthesis: automatic aggregations across related tables, with cutoff times | Yes, deterministic | **Use its idea (or library) as our non-LLM baseline**: "DFS features" vs "LLM SQL features" is an honest comparison |
| getML community (ELv2) | Fast relational feature learning (FastProp) | Yes | Community edition "must not be used for productive purposes"; best algorithms are paid. Shows the gap: no free production-grade option |
| TabPFN v2/2.5, **TabICL v2**, Mitra | Tabular foundation models; strong on small/medium data with no tuning | No | TabICL v2 is fully open; usable as an optional model. Note: RealTabPFN-2.5 is non-commercial |

### 3.2 Paid / platform tools

| Tool | What it does | Gap we can use |
|---|---|---|
| **Kumo / KumoRFM (now NVIDIA)** | Predictions directly over relational warehouses; "predictive query language"; foundation model | Proprietary, API/NIM-only; documented connectors Snowflake, Databricks, DuckDB, SQLite, S3 (no Postgres listed). **Adopt the idea of a declarative prediction task spec**, written our own way (open YAML, not their PQL syntax) |
| DataRobot (incl. Feature Discovery) | Enterprise AutoML, multi-table features, governance | Expensive, closed |
| Databricks AutoML + Data Science Agent | Notebook agent inside Databricks (preview) | Locked to Databricks |
| Google BigQuery ML + Colab/BigQuery Data Science Agent | Agent writes notebooks over BigQuery tables | Locked to Google Cloud; general-purpose, no leakage guarantees |
| SageMaker Autopilot/Canvas, Vertex AI tabular | Managed AutoML | Single table, cloud-locked |
| Julius, ChatGPT data analysis | Chat with a CSV | Single file, no guarantees, data goes to their servers |
| Feature stores (Feast, Tecton, Databricks) | Point-in-time joins **once you've defined features** | They don't invent features or the task |

### 3.3 Agentic data-science research projects

| Project | What it shows | Relevance |
|---|---|---|
| AIDE (Weco) | Tree search over code solutions; strong on MLE-bench | Single-table Kaggle setting |
| MLE-STAR (Google, NeurIPS 2025) | Web search + targeted refinement of code blocks; 63% medals on MLE-bench Lite | Same: assumes prepared files |
| DS-Agent, Data Interpreter, AutoKaggle, SELA | Case-based reasoning, plan graphs, multi-agent Kaggle pipelines | Same |
| **RelAgent (NeurIPS 2026)** | LLM agent writes **SQL feature programs** + picks a classical model; enforces "only data before t"; best on RelBench v2 | **Closest prior work. Our product is "RelAgent made usable on a live company DB, with guarantees, a UI and evidence."** Cite it everywhere |
| Rel-LLM; Wydmuch et al. (Snowflake AI Research) | LLMs reading related rows as documents for prediction | Alternative approach; expensive per prediction |
| AgentDS benchmark (2026) | AI-only agents score below top humans; **human+AI teams do best** on domain-specific data science | Supports our human-in-the-loop design |

### 3.4 Ideas worth adopting (all are ideas or permissive-licence libraries, not copied code)

1. **Task spec** as a declarative object (Kumo PQL idea) → our own YAML schema.
2. **Features as SQL** (RelAgent) → readable, run in production, no training/serving skew.
3. **Cutoff-time aggregations** (Featuretools DFS) → deterministic baseline and fallback.
4. **Evolutionary/feedback-driven proposals** (LLM-FE, CAAFE) → the loop shows the LLM which features helped.
5. **Search over multiple rollouts, pick by validation** (RelAgent, AIDE) → optional "N rollouts" setting with a cost cap.
6. **Static leakage checks on code** (Yang et al., LeakageDetector 2.0) → apply the same idea to our SQL: check every feature query's AST for cutoff violations.
7. **Experiment tracking export** (MLflow) → optional export so teams can keep using their tools.
8. **dbt models as output** → analytics teams already run dbt; a feature delivered as a dbt model fits how they work.
9. **MCP server** → lets Claude/Cursor/other agents call MLPilot as a tool ("for humans and agents", which Kumo now markets).

---

## 4. Research papers worth knowing

| Paper | Link | Idea for MLPilot |
|---|---|---|
| RelAgent: LLM Agents as Data Scientists for Relational Learning (Huang et al., NeurIPS 2026) | [arXiv 2605.07840](https://arxiv.org/abs/2605.07840), [code](https://github.com/HxyScotthuang/RelAgent) | Core method: SQL feature programs + classical model; deterministic inference |
| RelBench (Robinson et al., NeurIPS 2024) | [arXiv 2407.20060](https://arxiv.org/abs/2407.20060), [code](https://github.com/snap-stanford/relbench) | Public benchmark with temporal splits; our proof |
| RelBench v2 (2026) | [arXiv 2602.12606](https://arxiv.org/abs/2602.12606) | 11 databases incl. ERP (rel-salt) and clinical; autocomplete tasks |
| Position: Relational Deep Learning (Fey et al., 2024) | [arXiv 2312.04615](https://arxiv.org/abs/2312.04615) | Why multi-table ML matters; the GNN baseline |
| KumoRFM-2 (2026) | [arXiv 2604.12596](https://arxiv.org/abs/2604.12596) | The proprietary state of the art to compare with |
| Tackling prediction tasks in relational databases with LLMs (Wydmuch et al., 2024) | [arXiv 2411.11829](https://arxiv.org/abs/2411.11829) | LLM-only alternative; small models do well with good context |
| Rel-LLM (2025) | [arXiv 2506.05725](https://arxiv.org/abs/2506.05725) | Another LLM relational approach |
| 4DBInfer (Wang et al., 2024) | [arXiv 2404.18209](https://arxiv.org/abs/2404.18209) | Second relational benchmark (includes DFS baselines) |
| Deep Feature Synthesis (Kanter & Veeramachaneni, DSAA 2015) | [Featuretools](https://github.com/alteryx/featuretools) | Deterministic multi-table features with cutoff times |
| CAAFE (Hollmann, Müller, Hutter, NeurIPS 2023) | [arXiv 2305.03403](https://arxiv.org/abs/2305.03403) | LLM proposes features from column semantics, keeps those that improve CV |
| LLM-FE (2025) | [arXiv 2503.14434](https://arxiv.org/abs/2503.14434) | LLM as evolutionary optimizer of features, with feedback |
| ELF-Gym (2024) | [arXiv 2410.12865](https://arxiv.org/abs/2410.12865) | Benchmark of LLM-generated features vs Kaggle winners' features |
| FeatLLM (ICML 2024) | [arXiv 2404.09491](https://arxiv.org/abs/2404.09491) | LLM writes rules once, then cheap inference |
| Leakage and the reproducibility crisis in ML-based science (Kapoor & Narayanan, Patterns 2023) | [arXiv 2207.07048](https://arxiv.org/abs/2207.07048) | Leakage taxonomy (8 types) found across 294 papers in 17 fields; use for our leakage categories |
| Data Leakage in Notebooks: Static Detection (Yang et al., ASE 2022) | [arXiv 2209.03345](https://arxiv.org/abs/2209.03345) | Static analysis for leakage; apply to SQL |
| LeakageDetector 2.0 (2025) | [arXiv 2509.15971](https://arxiv.org/abs/2509.15971) | Tooling for leakage detection in notebooks |
| AIDE (2025) | [arXiv 2502.13138](https://arxiv.org/abs/2502.13138) | Tree search over solutions |
| MLE-STAR (NeurIPS 2025) | [arXiv 2506.15692](https://arxiv.org/abs/2506.15692) | Targeted refinement of pipeline blocks |
| MLZero / AutoGluon Assistant (2025) | [arXiv 2505.13941](https://arxiv.org/abs/2505.13941) | Multi-agent zero-code ML |
| MLE-bench (OpenAI, 2024) | [arXiv 2410.07095](https://arxiv.org/abs/2410.07095) | Standard agent benchmark (single-table) |
| AgentDS (2026) | [arXiv 2603.19005](https://arxiv.org/abs/2603.19005) | Human+AI beats AI-only on domain tasks |
| TabPFN (Hollmann et al., Nature 2025) / TabICL v2 | [TabPFN](https://github.com/PriorLabs/TabPFN), [TabICL](https://github.com/soda-inria/tabicl) | Optional strong models; mind TabPFN-2.5's non-commercial licence |

**What isn't in a usable product yet (the opportunity):** RelAgent's method + guaranteed point-in-time correctness + read-only safety on a live database + human review + evidence report + exportable SQL/dbt. Each piece exists somewhere; nobody ships them together, open source.

---

## 5. Recommended direction and USP

### 5.1 Positioning

- **One line:** *"Ask your database a prediction question. Get a leakage-safe model and the SQL to run it, all open-source and inside your own network."*
- **For engineers:** "An open-source, self-hosted agent that turns a relational database and a prediction question into point-in-time-correct training data, LLM-proposed SQL features that are validated by statistics, and an auditable evidence report. Inspired by RelAgent; compared publicly on RelBench."
- **Primary goal (headline):** solves a step others leave to humans: getting from many live tables to a correct, leak-free training set and features you can read.
- **Backup floor:** close to the paid tools' accuracy for a fraction of the cost, self-hosted, no lock-in.

### 5.2 The USP stack (what's true once built, in order of importance)

1. **Point-in-time guarantee enforced by a SQL compiler, not a prompt.** Every feature query is parsed (sqlglot) and rewritten so it can only read rows before the cutoff; a query that can't be proven safe is rejected. Planted-leak "canary" tests prove it in CI.
2. **Features you can read and run in production.** Each feature is a SQL query with its reason, measured gain and confidence interval. Export as dbt models or plain SQL.
3. **Read-only by construction.** Read-only DB role check + read-only transaction + statement allowlist + timeouts + row limits. The LLM sees schema and summary stats by default, never raw rows. Optional local LLM (Ollama) so nothing leaves the network.
4. **Honest statistics.** Temporal validation, nested tuning, accept only if the improvement beats noise, cumulative feature set, one final test touched once.
5. **Human-in-the-loop where it matters.** Human confirms the task spec and can veto features; the chat answers only from recorded run data.
6. **Public, reproducible benchmark** on RelBench with cost per task.

### 5.3 Who would use it (realistic)

- Small and mid-sized companies with a Postgres/MySQL product database and no ML team (SaaS, e-commerce, fintech, logistics), who want churn, conversion, late-payment or demand predictions.
- Freelancers and consultants doing those projects; MLPilot cuts their delivery time.
- Data/analytics engineers who know SQL and dbt but not ML.
- Researchers who want an open RelAgent-style baseline.

Not large enterprises with MLOps teams (they buy Kumo/Databricks).

---

## 6. Target architecture

```
          ┌──────────── UI (React) ─────────────┐   ┌─ CLI / MCP server ─┐
          │ Connect DB · Task builder · Schema   │   │  mlpilot run ...   │
          │ Feature review · Evidence · Chat     │   └─────────┬──────────┘
          └───────────────┬──────────────────────┘             │
                          ▼                                    ▼
                    FastAPI (backend/app)  ◄──────────── same core library
                          │
   ┌──────────────┬───────┴───────┬────────────────┬──────────────────┐
   ▼              ▼               ▼                ▼                  ▼
Connectors    Task spec       Training-table   Feature agent      Validation &
(read-only)   (YAML; LLM      builder          (LLM proposes SQL;  evidence
Postgres,     drafts, human   (labels at       sqlglot whitelist + (temporal CV,
MySQL,        confirms)       cutoffs, DuckDB  cutoff rewrite;     nested tuning,
SQLite,                       engine)          DFS baseline)       leakage canaries,
DuckDB, CSV                                                         report, export)
   │                                                                    │
   └──────── LLM gateway (existing: providers, fallback, cost) ─────────┘
```

Design defaults (each can be revisited through an ADR, see #95):

- **DuckDB is the internal engine.** CSV/Parquet uploads become a one-table DuckDB database, so there's one code path. Live databases are queried read-only, or snapshotted into DuckDB with a row limit for speed.
- **First connectors: PostgreSQL, SQLite, DuckDB, CSV/Parquet.** MySQL next; Snowflake/BigQuery later.
- **Default model: LightGBM** (fast, permissive licence). Optional engines: AutoGluon, TabICL v2.
- **LLM context = schema + column stats + a handful of sampled values per column only when the user allows it.** Default off.
- **Task spec format:** YAML with `entity`, `entity_key`, `target` (SQL expression over the future window), `horizon`, `cutoffs` (schedule), `filters`, `metric`.
- **Licence:** Apache-2.0 (there's no LICENSE file today).
- **Frontend:** keep `frontend/` (the one `start.py` runs), delete `frontend/new_ui/`.

---

## 7. Phased roadmap

Each phase has an exit test. Don't start a phase until the previous exit test passes on CI.

### Phase 0: Make it run, and make it honest (P0)
Fix the install, the start-up, the broken calls and the fake data. Add CI with an end-to-end test.
**Exit:** fresh clone → `pip install` → `python start.py` → sample dataset runs baseline + 2 experiments in the UI with real backend numbers; no mock data shown unless a "Demo mode" banner says so; CI green.

### Phase 1: Honest ML core (P1)
Validation split / nested tuning, CV-based acceptance with a margin, cumulative features, reject invalid formulas, deterministic keep/reject, leakage detector rewrite with a planted-leak test set, dataset versioning, a better AST evaluator, LightGBM default engine.
**Exit:** on the telecom sample and 3 OpenML datasets, reported metrics come from a test split never used for selection; leakage detector precision/recall on the planted-leak suite is printed in CI.

### Phase 2: Database connection done properly (P1)
Connection manager with secrets handling, read-only enforcement in depth, schema + relationship introspection (declared and inferred keys), DuckDB unification, the Connect Database UI with schema graph, demo Postgres database in docker-compose.
**Exit:** user connects to the demo Postgres from the UI, sees the schema graph, and any write/DDL statement is refused by three independent layers (tested).

### Phase 3: Prediction tasks and point-in-time training tables (P1)
Task spec schema, NL → spec via LLM with human confirmation, deterministic label generation, temporal splits, the cutoff-enforcing SQL rewriter, leakage canary tests.
**Exit:** on the demo DB, "customers who won't order in the next 30 days" produces a training table whose canary feature (planted future column) is blocked; tests prove every feature query respects the cutoff.

### Phase 4: SQL feature agent (P1/P2)
DFS-style deterministic baseline features; LLM proposes SQL aggregation features one at a time from history; whitelist + rewrite + execute; temporal-CV acceptance; budget and cost cap; human veto/approve in chat; optional N rollouts.
**Exit:** on the demo DB and on RelBench `rel-f1`, the agent improves validation AUROC over the DFS baseline (or honestly reports that it doesn't), within a stated LLM budget.

### Phase 5: Evidence report and export (P2)
Per-feature evidence (SQL, rationale, gain with CI, leakage checks passed), per-run report (HTML/Markdown), export as dbt project / SQL + model + scoring script, batch scoring CLI, real SHAP on the final model, optional MLflow logging, grounded chat Q&A and debrief.
**Exit:** an exported bundle runs on a clean machine against the same DB and reproduces the reported validation metric.

### Phase 6: Public benchmark (P2)
RelBench harness (start with `rel-f1`, `rel-trial`, `rel-avito` or others small enough for a laptop), compare: LightGBM on entity table only, DFS + LightGBM, MLPilot (cheap LLM), MLPilot (strong LLM), and published RDL / RelAgent / KumoRFM numbers; report cost and time per task.
**Exit:** a reproducible `make benchmark` and a table in the README, including the tasks where we lose.

### Phase 7: Distribution (P2/P3)
`pip install mlpilot` CLI, one-command Docker, local LLM via Ollama, MCP server, docs site, hosted demo on Hugging Face Spaces with the demo DB, 2-minute video.
**Exit:** someone outside the project runs it on their own Postgres from the README alone.

### Later (P3, only after Phase 6)
Write predictions back to a separate schema (opt-in), scheduled re-scoring, simple drift comparison between runs, Snowflake/BigQuery/MySQL connectors, team sharing/comments, regression targets and multi-class, recommendation-style tasks.

### What to delete or freeze now
- `frontend/new_ui/` (duplicate), Deployments/drift mock screens, `Math.random()` stats, `simulateMetrics()`, `mock.ts` replies (keep only behind an explicit Demo mode).
- Fake SHAP in `chat.py`.
- DataClean adapter (`ml/data/preparation/dataclean/`), since the project decided against it.
- Claims in README that aren't true yet.

---

## 9. Risks and how we'll know early

| Risk | How we'll find out | Cut-off decision |
|---|---|---|
| Cheap LLMs write poor SQL features | Phase 4 exit test on `rel-f1` vs DFS baseline | If the LLM can't beat DFS with a cheap model, ship DFS + LLM explanations and say so honestly |
| Point-in-time rewriting is harder than expected for complex SQL | Phase 3 tests | Restrict the LLM to a feature template grammar (aggregation × window × filter) instead of free SQL. Safer anyway |
| Companies won't connect a tool to production | Feedback from 3 trial users | Snapshot mode: export with a script, run on a replica or a dump |
| Kumo/NVIDIA or Databricks ship a free equivalent | Watch their releases | Our open-source, self-hosted, readable-SQL angle still differs; keep RelBench numbers current |
| Scope creep (again) | Issues per phase; nothing from phase N+1 starts early | Enforced through the issue labels |

---

## 12. Sources

Market and products
- Fortune, "Exclusive: Nvidia snaps up Kumo AI" (2026-06-03): https://fortune.com/2026/06/03/nvidia-snaps-up-kumo-ai-in-latest-acquisition/
- Forbes, "Nvidia Buys Kumo AI To Take Foundation Models To Enterprise Data": https://www.forbes.com/sites/janakirammsv/2026/06/10/nvidia-buys-kumo-ai-to-take-foundation-models-to-enterprise-data/
- GuruFocus, "Nvidia acquires Kumo AI for over $400M": https://www.gurufocus.com/news/8901384/nvidia-nvda-acquires-kumo-ai-for-over-400m
- NVIDIA Kumo Relational docs (connectors, API-only access): https://docs.nvidia.com/sdgm/rfm/overview
- Kumo, "Introducing KumoRFM": https://kumo.ai/company/news/kumo-relational-foundation-model/
- getML community (ELv2; community "must not be used for productive purposes"): https://github.com/getml/getml-community
- Featuretools: https://github.com/alteryx/featuretools
- AutoGluon Assistant: https://github.com/autogluon/autogluon-assistant ; Amazon Science blog: https://www.amazon.science/blog/autogluon-assistant-zero-code-automl-through-multiagent-collaboration
- AutoGluon tabular foundation models: https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html
- The state of tabular foundation models (2026): https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models
- Databricks Data Science Agent (InfoWorld): https://www.infoworld.com/article/4052376/databricks-adds-data-science-agent-to-automate-analytics-tasks.html
- Colab Enterprise Data Science Agent with BigQuery: https://docs.cloud.google.com/bigquery/docs/colab-data-science-agent
- Google Research, MLE-STAR: https://research.google/blog/mle-star-a-state-of-the-art-machine-learning-engineering-agents/
- Databricks point-in-time feature joins: https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series
- dbt + ML (Xebia): https://xebia.com/blog/dbt-machine-learning/

Papers: see the table in section 4.

---
