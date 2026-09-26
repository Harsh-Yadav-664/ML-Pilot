# MLPilot — Full Product Requirements Document

**Working name:** MLPilot  
**Product category:** Agentic ML experimentation and model-development platform  
**Document status:** Product blueprint / build specification  
**Primary goal:** Build a deployable, sellable product that helps non-specialist developers, students, researchers, freelancers, and small teams turn datasets + prediction goals into evidence-backed ML experiments, a reproducible model, and an understandable decision trail.

---

## 1. Executive Summary

MLPilot is not intended to be another black-box AutoML button.

The core product is an **AI-assisted experimentation system** that behaves like a junior ML/data scientist:

1. Understand the dataset and objective.
2. Profile data deterministically.
3. Identify validation, leakage, data-quality, and modeling risks.
4. Establish a reproducible baseline.
5. Generate ML hypotheses using an LLM.
6. Convert hypotheses into executable experiments.
7. Run those experiments through deterministic Python tooling.
8. Compare results against an explicit objective.
9. Decide what experiment should happen next.
10. Preserve the evidence and reasoning behind every decision.
11. Produce a final model, model card, training pipeline, prediction interface, and experiment history.
12. Optionally monitor future data/model performance and trigger new investigation cycles.

The defining product loop is:

> **Hypothesis → Experiment → Evidence → Decision → Next Hypothesis**

The LLM is not trusted to calculate metrics or silently modify the training process. Deterministic software owns data processing, validation, model training, metrics, constraints, and artifacts. The agent interprets evidence, proposes hypotheses, selects among allowed actions, and explains decisions.

This separation is central to product trustworthiness.

---

# 2. Problem

## 2.1 The actual user problem

A large number of developers and small teams can obtain a dataset but cannot efficiently answer:

- Is this dataset actually usable?
- What is wrong with it?
- Is there target leakage?
- What validation strategy should I use?
- What should I try first?
- Which features are worth engineering?
- Which models are worth testing?
- Why did one experiment improve the metric?
- Which experiment should I run next?
- Can I reproduce the result?
- Can I explain the model to another engineer/client?
- What should happen after deployment?

Existing tooling is fragmented.

A user may need a combination of:

- pandas / scikit-learn
- AutoML
- notebooks
- feature engineering code
- MLflow
- experiment tracking
- model explainability
- LLM APIs
- deployment tooling
- custom scripts

The problem is therefore not simply:

> "How do I train a model?"

It is:

> **"How do I systematically investigate an ML problem and turn experiments into a defensible model?"**

---

# 3. Product Vision

## Vision

> **Make evidence-driven ML experimentation accessible without requiring a dedicated data scientist.**

MLPilot should eventually feel like:

> "Give me the dataset, objective, constraints, and business context. I will investigate the modeling problem, run controlled experiments, show you the evidence, and give you a reproducible result."

The product must not hide experimentation behind a single score.

It should make the path from:

**raw data → hypothesis → experiment → evidence → decision → model**

visible and reproducible.

---

# 4. Positioning

Do not position the product primarily as "AutoML."

AutoML already covers automated model selection/search very well. Existing systems such as AutoGluon and FLAML provide automated modeling and feature-engineering capabilities, while MLflow provides experiment tracking, model lifecycle tooling, and increasingly sophisticated GenAI evaluation/observability. citeturn0search0turn0search5turn0search6turn0search8

MLPilot should instead position around:

### Primary category

**Agentic ML Experimentation**

### Secondary descriptions

- AI-assisted ML development
- Evidence-driven model experimentation
- AI data-science experimentation assistant
- ML investigation and decision platform

### Core pitch

> **MLPilot doesn't just search for a model. It investigates the ML problem, runs controlled experiments, remembers what happened, and explains what to try next.**

---

# 5. Differentiation / USP Strategy

We should not claim that every individual capability is globally unique. The differentiation should come from the **combination and implementation quality**.

## USP 1 — Evidence-driven experiment loop

Traditional automation often optimizes the search space.

MLPilot makes the unit of progress an **experiment with a hypothesis**.

Example:

> Hypothesis: customer inactivity predicts churn.

The system creates:

> `days_since_last_purchase`

Then runs:

> Baseline vs baseline + feature

Then records:

- metric delta
- validation evidence
- statistical/stability information where appropriate
- cost
- runtime
- decision
- rationale

This creates an **auditable scientific workflow**, rather than a pile of model scores.

---

## USP 2 — Decision provenance / Experiment Memory

Every meaningful decision has lineage.

For example:

```text
Experiment #27
│
├── Parent: Experiment #21
├── Hypothesis: Recency is predictive
├── Change: + days_since_last_purchase
├── Model: CatBoost
├── Validation: 5-fold stratified CV
├── F1: 0.798 → 0.821
├── Recall: 0.825 → 0.847
├── Cost: $0.00
├── Runtime: 41 sec
├── Decision: KEEP
└── Reason:
    Improved primary metric and recall without violating constraints.
```

The user can therefore answer:

> "Why is this feature in the final model?"

without reverse-engineering notebooks.

---

## USP 3 — Leakage-aware experimentation

MLPilot should treat leakage prevention as a first-class product capability.

It should detect and flag:

- post-target features
- target-derived columns
- suspicious timestamps
- duplicate/near-duplicate leakage
- train/test contamination
- entity leakage
- preprocessing leakage
- temporal leakage
- overly optimistic validation
- suspiciously perfect metrics

The system should never silently delete a feature.

It should show:

> **Why this feature is suspicious**

and require an explicit policy/decision where appropriate.

---

## USP 4 — Cost-aware intelligence

The LLM is not the experiment engine.

MLPilot should use different models/providers according to task complexity.

Example:

| Task | Preferred tier |
|---|---|
| Format structured metadata | cheap/free model |
| Summarize metrics | cheap/free model |
| Generate feature hypotheses | medium model |
| Resolve conflicting experiment evidence | stronger model |
| Generate final report | medium model |
| Critical decision review | configurable stronger model |

This reduces API costs and creates provider independence.

---

## USP 5 — Model/provider independence

The product should never architect itself around one LLM vendor.

The AI layer should support:

- OpenAI
- Anthropic
- Google Gemini
- Groq-hosted models
- OpenRouter-compatible providers
- local models where practical
- future providers

The provider router should support:

- capability routing
- cost routing
- latency routing
- fallback
- rate-limit handling
- model allowlists
- BYOK
- usage budgets

This is an architectural requirement, not merely a UI feature.

---

# 6. Target Customers

## Primary

### A. Indie developers

Developers building:

- churn prediction
- fraud detection
- lead scoring
- demand prediction
- recommendation systems
- classification systems
- regression systems

They have data but limited ML specialization.

### B. Small startups

Teams with engineers but no dedicated data scientist.

### C. Students / researchers

Users who need systematic experimentation and reproducibility.

### D. Freelancers / consultants

People doing ML prototypes for clients.

---

## Secondary

- educational institutions
- bootcamps
- research labs
- internal analytics teams
- product teams
- ML engineering teams that want an experiment assistant

---

## Not initially targeted

Large enterprises already operating mature ML platforms.

They can become an enterprise segment later, but the initial product should not attempt to compete directly with the full platforms offered by Dataiku, DataRobot, Databricks, etc.

Dataiku, for example, already combines visual ML/AutoML, deployment, governance, model lifecycle, and LLM-provider routing. citeturn0search10

MLPilot should win a narrower problem first:

> **Intelligent experimentation and decision-making.**

---

# 7. User Personas

## Persona 1 — Developer

"I know Python and can use pandas, but I don't want to spend three days testing random models."

Needs:

- fast experimentation
- sensible defaults
- understandable explanations
- deployable artifacts

## Persona 2 — Student

"I need to build a serious ML project and explain every decision."

Needs:

- experiment history
- reports
- graphs
- reproducibility
- educational explanations

## Persona 3 — Startup engineer

"We need a model but don't have an ML specialist."

Needs:

- guardrails
- automated experimentation
- model packaging
- monitoring
- team collaboration

## Persona 4 — Freelancer

"I need to deliver a credible model and report to clients."

Needs:

- client-ready reports
- reproducibility
- export
- experiment evidence
- configurable branding in paid tiers

---

# 8. Product Principles

1. **Evidence over vibes.**
2. **Deterministic computation over LLM hallucination.**
3. **Every important experiment is reproducible.**
4. **The agent proposes; the experiment engine verifies.**
5. **The user controls objective and constraints.**
6. **Never optimize blindly against the test set.**
7. **No silent data destruction.**
8. **Provider independence.**
9. **Free-first architecture.**
10. **Everything important should be inspectable.**
11. **Every feature should have an exit path: export code/data/artifacts.**
12. **The system should be useful even if every LLM API is unavailable.**

---

# 9. Scope

## Supported initial task family

V1/V2:

- binary classification
- multiclass classification
- regression

Future:

- time-series forecasting
- ranking
- anomaly detection
- survival analysis
- multilabel classification
- NLP
- multimodal/tabular-text problems

Do not attempt all task types in V1.

---

# 10. End-to-End Product Workflow

```text
User
  ↓
Project creation
  ↓
Dataset ingestion
  ↓
Objective definition
  ↓
Data profiling
  ↓
Data readiness
  ↓
Leakage / validation analysis
  ↓
Baseline
  ↓
Semantic data understanding
  ↓
Hypothesis generation
  ↓
Experiment planner
  ↓
Experiment execution
  ↓
Evidence evaluation
  ↓
Decision
  ↓
Next experiment
  ↓
Repeat until stopping criteria
  ↓
Final validation
  ↓
Model selection
  ↓
Explainability
  ↓
Packaging
  ↓
Deployment
  ↓
Monitoring
  ↓
Future experiment/retraining loop
```

---

# 11. Functional Requirements

## 11.1 Project creation

A project contains:

- project name
- objective
- task type
- target column
- primary metric
- secondary metrics
- constraints
- dataset versions
- experiments
- models
- reports
- deployment environments

---

## 11.2 Objective configuration

User should specify:

- target
- task
- optimization metric
- metric direction
- minimum acceptable threshold
- class priority where relevant
- maximum experiment budget
- maximum runtime
- API/compute budget
- interpretability preference
- latency requirement
- model size constraint

Example:

```yaml
objective:
  task: classification
  target: churned
  primary_metric: f1
  secondary_metrics:
    - recall
    - precision
  constraints:
    recall_min: 0.80
    max_runtime_seconds: 300
    max_llm_cost_usd: 0.50
```

---

# 12. Dataset Ingestion

Initial:

- CSV
- Parquet

Future:

- Excel
- SQL databases
- PostgreSQL
- MySQL
- cloud object storage
- APIs
- data warehouses

Requirements:

- schema detection
- encoding detection
- type inference
- size limits
- sampling for large datasets
- dataset versioning
- checksum/fingerprint
- secure temporary storage

---

# 13. Data Profiling Engine

Deterministic Python.

Detect:

- row/column count
- data types
- missing values
- duplicates
- unique counts
- constant columns
- near-constant columns
- high-cardinality columns
- ID-like columns
- date/time columns
- distributions
- outliers
- class imbalance
- correlations
- suspicious values
- target distribution

No LLM required.

Output:

```json
{
  "rows": 42831,
  "columns": 19,
  "missing_rate": 0.074,
  "duplicate_rows": 312,
  "target_balance": {
    "0": 0.82,
    "1": 0.18
  }
}
```

---

# 14. Data Preparation Architecture

The system must remain open to two implementation paths.

## Option A — Build our own preparation engine

Build:

- missing-value handling
- duplicate handling
- type normalization
- outlier policies
- encoding
- scaling
- date parsing
- leakage checks
- readiness scoring

Advantages:

- full control
- no external dependency
- easier product ownership
- consistent API

## Option B — Integrate friend's DataClean project

DataClean can become a preparation provider/module.

Architecture:

```text
MLPilot
   ↓
Data Preparation Interface
   ├── Native Preparation Engine
   └── DataClean Adapter
```

The rest of MLPilot must not know which implementation was used.

This is a hard architectural requirement.

### Interface example

```python
class DataPreparationProvider:
    def profile(self, dataset): ...
    def assess_readiness(self, dataset): ...
    def detect_leakage(self, dataset, target): ...
    def prepare(self, dataset, config): ...
```

This allows us to decide later.

### Important

Do not architect the product around the assumption that DataClean will be used.

The project must remain fully functional without it.

---

# 15. Semantic Data Understanding

The LLM receives structured metadata rather than blindly receiving the entire dataset.

Example:

```json
{
  "column": "last_purchase",
  "type": "datetime",
  "unique_ratio": 0.82,
  "sample_values": ["2026-01-04", "..."],
  "relationship_candidates": ["signup_date"]
}
```

The LLM can infer:

- likely semantic meaning
- potential relationships
- candidate transformations
- feature hypotheses
- risk explanations

The LLM should not directly mutate the dataset.

---

# 16. Leakage Detection

Multiple layers:

### Rule-based

- target column in feature set
- exact target duplicates
- post-target timestamps
- target-derived names
- impossible training-time information

### Statistical

- suspiciously high feature-target association
- train/test distribution anomalies
- duplicates across splits

### Temporal

- future information used for historical prediction

### Entity

- same entity appearing across train/test when inappropriate

### LLM-assisted

The LLM can explain why a suspicious feature may be problematic.

Final enforcement remains deterministic.

---

# 17. Validation Engine

Validation strategy should depend on task/data.

Possible strategies:

- stratified K-fold
- K-fold
- repeated K-fold
- group K-fold
- time-series split
- holdout
- nested validation where justified

The system should explain:

> "Why this validation strategy was selected."

Users can override it.

---

# 18. Baseline Engine

Baseline models should be deterministic and configurable.

Classification:

- Dummy classifier
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost/LightGBM/CatBoost where available

Regression:

- Dummy regressor
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost/LightGBM/CatBoost

The baseline establishes the starting point.

---

# 19. Feature Hypothesis Agent

The agent generates candidate features.

Each hypothesis must contain:

```json
{
  "name": "days_since_last_purchase",
  "formula": "prediction_date - last_purchase",
  "reason": "Recent inactivity may correlate with churn.",
  "risk": "Requires prediction-time availability.",
  "expected_effect": "Potential recall improvement"
}
```

The system then tests it.

The LLM cannot declare it successful.

---

# 20. Feature Experiment Engine

Experiments should be isolated.

Example:

```text
Baseline
  ↓
+ Feature A
  ↓
+ Feature B
  ↓
Feature A + Feature B
```

Avoid uncontrolled accumulation.

Every experiment records:

- parent experiment
- code/config version
- dataset version
- feature set
- preprocessing version
- model
- hyperparameters
- validation method
- metrics
- runtime
- resource usage
- LLM cost
- decision
- rationale

---

# 21. Model Experimentation

Models should be selected according to:

- task
- dataset size
- feature types
- constraints
- compute budget
- interpretability needs

The system should not always run every model.

The agent should narrow the search based on evidence.

---

# 22. Agentic Experiment Planner

This is the central intelligence.

Input:

- experiment history
- objective
- constraints
- dataset findings
- model results
- feature results
- rejected branches
- compute/API budget

Output:

```json
{
  "action": "feature_experiment",
  "hypothesis": "...",
  "candidate_features": [...],
  "model": "catboost",
  "reason": "...",
  "expected_information_gain": 0.72
}
```

The planner should optimize not only expected performance but also:

- information gain
- cost
- runtime
- uncertainty reduction
- risk
- diversity of search

---

# 23. Experiment Selection Strategy

Future versions should support an experiment utility function:

```text
Utility =
expected improvement
+ information gain
+ uncertainty reduction
- compute cost
- API cost
- redundancy penalty
- risk penalty
```

This is a major area for research and differentiation.

---

# 24. Experiment Memory

Persist:

- experiment graph
- hypotheses
- results
- decisions
- failed experiments
- rejected features
- selected models
- model versions
- prompts
- LLM provider/model
- costs
- timestamps
- dataset versions

The agent must know what it already tried.

---

# 25. Experiment Graph

UI should show:

```text
                         Baseline
                            |
              +-------------+-------------+
              |             |             |
           Feature A     Feature B     Model B
              |             X             |
           accepted                     rejected
              |
           CatBoost
              |
        +-----+------+
        |            |
      Tuning       Feature C
        |            X
        |
      Final
```

Users should be able to click any node and inspect the full evidence.

---

# 26. AI Evaluator

Python calculates:

- metrics
- confidence intervals where appropriate
- fold variance
- statistical tests where justified

LLM interprets:

- what changed
- whether it matters
- tradeoffs
- likely reasons
- what to try next

The LLM should never fabricate metrics.

All numerical claims shown in AI explanations must reference stored experiment results.

---

# 27. Stopping Criteria

The agent must know when to stop.

Possible rules:

- improvement below threshold for N experiments
- budget exhausted
- time limit reached
- confidence/stability threshold reached
- objective threshold achieved
- search space exhausted
- diminishing information gain
- validation instability
- no meaningful candidate experiments remaining

Output:

> "Stopping because the last 6 valid experiments produced less than 0.5% improvement and the configured objective has been satisfied."

---

# 28. Model Selection

Selection should consider:

1. primary metric
2. required constraints
3. secondary metrics
4. fold stability
5. model complexity
6. inference latency
7. memory footprint
8. interpretability
9. reproducibility
10. operational cost

The user should be able to configure priorities.

---

# 29. Explainability

Support:

- feature importance
- permutation importance
- SHAP where practical
- confusion matrix
- ROC curve
- PR curve
- residual analysis
- subgroup analysis

The system should distinguish:

> predictive association

from:

> causal explanation.

---

# 30. Model Card

Generate:

- model purpose
- training data
- target
- features
- preprocessing
- validation strategy
- metrics
- limitations
- known risks
- intended use
- non-intended use
- explainability
- experiment lineage
- version
- training date

---

# 31. Reproducible Export

Export:

```text
mlpilot-export/
├── model.joblib
├── preprocessing/
├── train.py
├── predict.py
├── requirements.txt
├── model_card.md
├── experiment_history.json
├── experiment_graph.json
├── metrics.json
├── feature_definitions.json
└── README.md
```

Optional:

- Dockerfile
- FastAPI server
- batch prediction script
- OpenAPI spec

---

# 32. Deployment

V4+.

One-click deployment options:

### A. Hosted API

```text
POST /predict
```

### B. Docker export

```bash
docker build .
docker run ...
```

### C. Serverless deployment

Future adapters:

- Azure
- AWS
- GCP
- Railway
- Render
- Fly.io

### D. Local deployment

For privacy-sensitive users.

---

# 33. Monitoring

Post-deployment:

- prediction volume
- latency
- error rate
- data drift
- feature drift
- target drift when labels arrive
- model performance
- class distribution
- missing-value changes

Future:

> Model degradation → investigation → new experiment cycle.

This closes the loop.

---

# 34. Continuous Experimentation

Future product loop:

```text
Production
   ↓
Monitoring
   ↓
Drift / performance issue
   ↓
Agent investigation
   ↓
New hypothesis
   ↓
Controlled experiment
   ↓
Candidate model
   ↓
Approval
   ↓
Deployment
```

This turns MLPilot from a one-time model builder into an ML lifecycle product.

---

# 35. Multi-LLM Provider Architecture

## Provider abstraction

```text
AI Gateway
│
├── OpenAI
├── Anthropic
├── Gemini
├── Groq
├── OpenRouter
├── Local/Ollama
└── Future providers
```

Each provider exposes:

- model
- capabilities
- context window
- cost
- latency
- availability
- rate limits

---

# 36. AI Router

Router chooses model using:

```text
task complexity
+ required capability
+ cost budget
+ latency requirement
+ provider availability
+ user preference
```

Fallback:

```text
Primary provider
   ↓ failure
Secondary provider
   ↓ failure
Local/basic model
   ↓ failure
Deterministic fallback
```

---

# 37. Free API Strategy

Initial product should be usable with:

- free-tier LLM APIs
- user-provided API keys
- optional local models
- deterministic functionality without LLM

Do not make the product economically dependent on subsidized free APIs.

Production architecture should support:

### BYOK

Users bring their own API keys.

### Platform-managed keys

Paid plans can use MLPilot-managed provider accounts.

### Local/private mode

Users can use local models where supported.

---

# 38. Cost Controls

Every project gets:

- LLM token budget
- maximum experiment budget
- maximum runtime
- model routing policy
- provider fallback policy

Dashboard:

```text
LLM usage
$0.07 / $1.00 budget

Experiments
17 / 50

Compute
8m 21s
```

---

# 39. Security Requirements

Must include:

- encrypted API keys
- secrets never logged
- project isolation
- temporary dataset storage
- automatic cleanup
- signed/downloadable artifacts
- authentication
- authorization
- rate limiting
- audit logs
- safe code execution
- container isolation
- resource limits
- upload validation
- malware/file scanning where hosted
- no arbitrary host execution

Most importantly:

### Never execute generated Python directly on the host.

Use isolated workers/containers/sandboxes.

---

# 40. Architecture

Recommended high-level architecture:

```text
                    Web App
                       |
                 API Gateway
                       |
              Project/Job Service
                       |
        +--------------+--------------+
        |                             |
   Experiment DB                 Object Storage
        |                             |
        +--------------+--------------+
                       |
                Orchestrator
                       |
       +---------------+---------------+
       |               |               |
 Data Engine     Experiment Engine   AI Gateway
       |               |               |
       |               |        +------+------+
       |               |        |             |
       |               |     Provider A   Provider B
       |               |
       +------- Worker Pool -------+
                       |
              ML Training Sandbox
                       |
             Artifacts / Metrics
                       |
                 Model Registry
                       |
                Deployment API
                       |
                  Monitoring
```

---

# 41. Suggested Technical Stack

## Frontend

- React
- TypeScript
- Vite/Next.js
- Tailwind
- charts/visualization

## Backend

Python is preferred because the ML ecosystem is Python-native.

- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL
- Redis

## ML

- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Optuna later

AutoGluon/FLAML can be optional experiment engines rather than the product itself.

## Experiment tracking

Initially:

- custom experiment schema

Potential integration:

- MLflow

MLflow already provides experiment/run tracking, artifacts, model metadata, and GenAI tracing/evaluation, so MLPilot should avoid rebuilding infrastructure that does not differentiate the product. citeturn0search6turn0search8

## Workers

- Docker
- Celery/RQ or an equivalent queue
- isolated worker containers

## Storage

- PostgreSQL
- S3-compatible object storage

## Auth

- OAuth
- email/password
- Google/GitHub login

---

# 42. Data Model

Core entities:

```text
User
Organization
Project
Dataset
DatasetVersion
Objective
Experiment
ExperimentArtifact
Hypothesis
Feature
Model
ModelVersion
Deployment
Prediction
MonitoringSnapshot
AIRequest
Provider
UsageRecord
Report
AuditEvent
```

---

# 43. Experiment Schema

Minimum:

```json
{
  "id": "...",
  "parent_id": "...",
  "project_id": "...",
  "dataset_version": "...",
  "hypothesis": "...",
  "change": "...",
  "model": "...",
  "parameters": {},
  "validation": {},
  "metrics": {},
  "artifacts": [],
  "runtime": 41.2,
  "cost": 0.00,
  "status": "completed",
  "decision": "keep",
  "decision_reason": "...",
  "agent_model": "...",
  "timestamp": "..."
}
```

---

# 44. UX / Main Screens

## Dashboard

- active projects
- best models
- experiment counts
- recent activity
- deployment health
- usage/cost

## Project overview

- objective
- dataset health
- current best model
- experiment progress
- agent recommendation

## Data page

- schema
- distributions
- missingness
- warnings
- leakage

## Experiment page

- hypothesis
- setup
- results
- comparison
- AI interpretation

## Experiment graph

Visual lineage.

## Agent page

Show:

> What I think

> Evidence

> What I want to try

> Expected benefit

> Cost

> Approve / Run

Depending on autonomy level.

## Model page

- metrics
- explainability
- limitations
- artifacts
- deployment

---

# 45. Agent Autonomy Modes

### Mode 1 — Suggest

Agent proposes experiments.

User approves.

### Mode 2 — Assisted

Agent runs low-risk experiments automatically.

### Mode 3 — Autonomous

Agent runs experiments within configured budget and constraints.

This should be a paid/pro feature eventually.

---

# 46. Human Approval Gates

Always configurable.

Require approval for:

- changing target
- changing validation strategy
- using suspicious features
- exceeding budget
- deploying production model
- replacing production model
- deleting artifacts
- accessing sensitive external data

---

# 47. Reports

Generate:

### Technical report

For ML engineers.

### Executive report

For non-technical stakeholders.

### Client report

For freelancers/consultants.

### Academic/research report

Future.

---

# 48. Service-Based Business Model

MLPilot can be sold in two forms.

## SaaS

Self-service platform.

### Free

- limited projects
- limited experiments
- BYOK
- local/basic models
- limited storage

### Pro

- more experiments
- hosted LLM usage
- autonomous mode
- deployment
- monitoring
- advanced reports

### Team

- collaboration
- shared projects
- organization-level provider keys
- audit logs
- roles
- higher limits

### Enterprise

Future:

- private deployment
- SSO
- VPC/on-prem
- custom models
- custom integrations
- governance

---

# 49. Service-Based Offering

A stronger early monetization path may be:

> **MLPilot-assisted ML development service**

Client provides:

- dataset
- objective
- business context

We deliver:

- data assessment
- experiment report
- selected model
- training pipeline
- model card
- deployment API
- monitoring setup

MLPilot becomes the internal engine powering the service.

This lets the product generate revenue before the SaaS product reaches scale.

---

# 50. Pricing Philosophy

Do not initially compete purely on "cheapest AI."

Charge for:

- compute
- convenience
- experimentation speed
- reproducibility
- deployment
- monitoring
- collaboration
- professional reports

Potential pricing can be tested later rather than hard-coded into V1.

---

# 51. What Makes the Product Sellable

The buyer should receive something concrete.

Not:

> "AI analyzed your CSV."

Instead:

```text
Dataset assessment
+
Experiment evidence
+
Best model
+
Why it was selected
+
Known limitations
+
Reproducible training code
+
Prediction API
+
Monitoring
```

That is an actual deliverable.

---

# 52. Future Feature Expansion

Architecture must remain open for:

## Data connectors

- SQL
- warehouses
- APIs
- cloud storage

## Advanced ML

- time series
- NLP
- computer vision
- multimodal
- anomaly detection

## Collaboration

- comments
- experiment approvals
- team workspaces

## Governance

- model approvals
- lineage
- audit logs
- policy enforcement

## AI research assistant

Ask:

> "Why did the model improve?"

> "What have we not tried?"

> "Which experiments were redundant?"

## Experiment recommendation marketplace

Future community-shared experiment strategies.

## Domain packs

Examples:

- SaaS churn
- fraud
- marketing
- demand forecasting
- credit risk
- cybersecurity

## Benchmark mode

Run the same objective across:

- model families
- feature strategies
- validation strategies
- LLM agents

---

# 53. Potential Moat

The long-term moat should not be "we call an LLM."

Potential moat:

### 1. Experiment corpus

Anonymized/opt-in historical experiment data.

### 2. Experiment recommendation intelligence

Learn which experiment types are useful under which dataset conditions.

### 3. Domain-specific experiment playbooks

### 4. Evaluation benchmark

Measure agent decisions against expert decisions.

### 5. Reproducible experiment graph

### 6. Cost/performance optimization

Learn when an expensive experiment is actually worth running.

---

# 54. Agent Evaluation

The agent itself must be evaluated.

Track:

- experiment usefulness
- redundant experiment rate
- invalid experiment rate
- leakage violations
- improvement per experiment
- cost per improvement
- time to objective
- stopping quality
- decision consistency

This is important because the product itself is an AI system.

MLflow demonstrates the broader importance of evaluating agent behavior using traces, datasets, and scorers; MLPilot should adopt similar principles where useful rather than treating the agent as an unmeasured black box. citeturn0search1turn0search2

---

# 55. Core Success Metrics

## Product metrics

- time to first valid baseline
- time to useful model
- experiments per project
- successful project completion rate
- user retention
- conversion to paid
- deployment rate

## ML metrics

- improvement over baseline
- leakage detection precision/recall
- experiment usefulness
- reproducibility rate
- model stability

## Agent metrics

- useful experiment rate
- redundant experiment rate
- invalid proposal rate
- average improvement per experiment
- cost per useful experiment

---

# 56. Non-Goals

At least initially:

- replacing expert ML engineers
- arbitrary code execution
- training huge foundation models
- building a general-purpose notebook
- building a full data warehouse
- becoming a complete enterprise data platform
- supporting every ML problem immediately

---

# 57. Risks

## Risk: "This is just AutoML"

Mitigation:

- experiment hypotheses
- experiment graph
- agentic next-step selection
- decision provenance
- evidence-based reports

## Risk: LLM makes bad decisions

Mitigation:

- deterministic engine
- action schemas
- constraints
- validation
- approval gates
- agent evaluation

## Risk: API costs

Mitigation:

- multi-provider routing
- BYOK
- caching
- small models
- budget limits
- local models

## Risk: competition

Mitigation:

- focus on experimentation intelligence
- narrow initial customer segment
- service-first revenue option
- build strong experiment lineage

## Risk: overbuilding

Mitigation:

- phased roadmap
- task-type restrictions
- V1 only tabular classification/regression

---

# 58. MVP Definition

The MVP is NOT the final product.

MVP should prove:

> Can an agent generate useful ML hypotheses and choose the next experiment better than blind/manual experimentation?

### MVP scope

- CSV upload
- target selection
- classification/regression
- deterministic profiling
- leakage warnings
- baseline
- 2–4 model families
- LLM feature hypothesis generation
- feature experiments
- experiment storage
- simple experiment graph
- agent recommendation for next experiment
- final model
- model card
- training script export

### Explicitly exclude from MVP

- multi-user organizations
- billing
- production deployment
- monitoring
- complex connectors
- every ML task
- enterprise security
- autonomous production deployment

---

# 59. V1 Definition

V1 becomes a serious portfolio/demo product.

Add:

- polished UI
- experiment comparison
- agent decision panel
- experiment graph
- provider abstraction
- BYOK
- cost tracking
- configurable objectives
- better leakage detection
- reproducible artifacts
- Docker export
- FastAPI prediction server
- technical report
- public demo dataset

---

# 60. V2 Definition

V2 becomes a usable hosted service.

Add:

- authentication
- cloud storage
- background workers
- job queue
- project persistence
- deployment
- model registry
- monitoring
- drift detection
- team workspaces
- usage limits
- paid plans

---

# 61. V3 Definition

V3 becomes a differentiated product.

Add:

- autonomous experiment mode
- experiment utility scoring
- cost-aware planning
- advanced experiment graph
- experiment templates
- domain playbooks
- advanced model routing
- provider fallback
- richer reports
- model replacement workflow
- experiment benchmarking

---

# 62. V4 Definition

Production/SaaS maturity.

Add:

- multi-tenancy
- billing
- RBAC
- audit logs
- SSO
- secure isolated compute
- enterprise deployment
- private provider configuration
- monitoring at scale
- SLA/observability
- data retention controls

---

# 63. V5 Definition

Long-term platform.

Potential:

- continuous ML experimentation
- automatic retraining proposals
- domain-specific agents
- experiment recommendation learned from prior projects
- team knowledge base
- marketplace
- research/benchmark platform
- API/SDK
- integrations

---

# 64. Build-vs-Integrate Policy

Every major subsystem should be classified:

### Build

If it is core differentiation.

Examples:

- experiment planner
- experiment graph
- hypothesis system
- decision provenance
- experiment utility scoring

### Integrate

If it is commodity infrastructure.

Examples:

- authentication
- object storage
- queue
- MLflow
- model libraries
- payment provider

### Adapter

If an external/internal project may replace our implementation.

Example:

- Data preparation

This keeps the friend's DataClean project optional.

---

# 65. Friend's DataClean Integration Decision

Do not decide at architecture stage.

Build the interface first.

Then benchmark:

```text
Native MLPilot preparation
vs
DataClean
```

Compare:

- quality
- leakage detection
- coverage
- runtime
- maintenance
- API quality
- license
- deployment complexity
- reproducibility

If DataClean is better, integrate it.

If our implementation is better, keep native.

If both are useful, expose both as preparation providers.

---

# 66. Open Architecture for Future Features

The platform should use plugin-style interfaces:

```text
DatasetProvider
DataPreparationProvider
FeatureGenerator
ExperimentRunner
ModelProvider
MetricProvider
ValidationStrategy
AIProvider
DeploymentProvider
MonitoringProvider
ReportProvider
```

This makes future expansion much easier.

---

# 67. Final Product Definition

The mature MLPilot product should be:

> **A deployable agentic ML experimentation platform that turns a dataset and objective into an evidence-backed, reproducible, deployable ML system while preserving the full reasoning and experiment history behind the result.**

The final workflow:

```text
Dataset
  ↓
Understand
  ↓
Validate
  ↓
Baseline
  ↓
Hypothesize
  ↓
Experiment
  ↓
Measure
  ↓
Remember
  ↓
Decide
  ↓
Experiment again
  ↓
Select
  ↓
Explain
  ↓
Package
  ↓
Deploy
  ↓
Monitor
  ↓
Investigate again
```

---

# 68. Product North Star

The north-star question for every feature:

> **Does this help the system make better ML decisions with stronger evidence, lower cost, or greater reproducibility?**

If not, it should probably not be in the core product.

