# MLPilot — Full Build Phases & Execution Roadmap

This roadmap covers the project from the first portfolio-quality MVP through a deployable and sellable product/service.

The roadmap intentionally keeps **DataClean optional**. The architecture must support both:

1. building the data-preparation layer ourselves, and
2. plugging in the friend's DataClean project later.

Do not make this decision prematurely.

---

# Phase 0 — Product Validation & Architecture

## Goal

Prove that the project is worth building and lock the boundaries before writing large amounts of code.

## Tasks

### Product

- Finalize problem statement.
- Define initial target users.
- Define the first supported tasks:
  - binary classification
  - multiclass classification
  - regression
- Define primary USP:
  - evidence-driven agentic experimentation.
- Define secondary differentiation:
  - experiment memory
  - decision provenance
  - leakage-aware experimentation
  - cost-aware AI routing.

### Technical

Design:

```text
Frontend
Backend
Experiment Orchestrator
ML Engine
AI Gateway
Worker Runtime
Database
Artifact Storage
```

Define interfaces:

```text
DataPreparationProvider
AIProvider
ExperimentRunner
ModelProvider
MetricProvider
ValidationStrategy
ReportProvider
DeploymentProvider
```

## Deliverables

- architecture diagram
- repository structure
- database schema draft
- experiment JSON schema
- provider interfaces
- product scope
- MVP acceptance criteria

## Exit criterion

You can replace:

```text
NativeDataPrep
```

with:

```text
DataCleanAdapter
```

without rewriting the experiment engine.

---

# Phase 1 — Deterministic ML Core

## Goal

Build the product without depending on an LLM.

This is critical.

If the AI layer disappeared, the ML system should still work.

## Build

### Dataset ingestion

- CSV
- schema inference
- type detection
- dataset fingerprint

### Profiling

- shape
- missingness
- duplicates
- cardinality
- distributions
- class balance
- outliers
- date detection
- ID detection

### Validation

- train/test split
- stratified CV
- standard K-fold
- time split where appropriate

### Baselines

Classification:

- Dummy
- Logistic Regression
- Random Forest
- Gradient Boosting

Regression:

- Dummy
- Linear Regression
- Random Forest
- Gradient Boosting

### Metrics

Classification:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

Regression:

- MAE
- MSE
- RMSE
- R²
- MAPE where appropriate

## Deliverables

A CLI or API that can do:

```text
dataset
→ profile
→ validate
→ baseline
→ compare
```

## Exit criterion

Given a CSV and target, the system produces a reproducible baseline and metrics without any LLM.

---

# Phase 2 — Data Readiness & Leakage Engine

## Goal

Make data quality and leakage a core product capability.

## Build

### Rule-based leakage detection

- target leakage
- target-derived features
- post-event columns
- suspicious timestamps
- train/test duplication
- preprocessing leakage
- entity leakage

### Data readiness

Produce:

```text
Data Quality Score
Leakage Risk
Validation Risk
Feature Risk
```

Do not collapse everything into one meaningless score; show the underlying evidence.

## Deliverables

Example:

```text
WARNING

cancellation_date

Potential post-target leakage.

Reason:
The value may only exist after the churn event.

Suggested action:
Exclude or verify availability at prediction time.
```

## Exit criterion

The engine catches deliberately planted leakage cases in a test suite.

---

# Phase 3 — Data Preparation Provider Layer

## Goal

Keep the DataClean decision open.

## Build native provider

Implement:

- missing-value handling
- categorical encoding
- scaling
- datetime processing
- duplicate policy
- basic outlier policy
- preprocessing pipeline generation

## Build interface

```python
class DataPreparationProvider:
    profile()
    assess_readiness()
    detect_leakage()
    prepare()
    export_pipeline()
```

## Optional integration

Create:

```text
DataCleanAdapter
```

but do not make it mandatory.

## Benchmark

Compare:

```text
MLPilot Native
DataClean
```

Against:

- correctness
- leakage detection
- runtime
- reproducibility
- API simplicity
- maintenance
- coverage

## Exit criterion

Either provider can power the system.

---

# Phase 4 — AI Gateway & Multi-Provider Layer

## Goal

Introduce LLMs without vendor lock-in.

## Build

```text
AI Gateway
├── OpenAI
├── Anthropic
├── Gemini
├── Groq
├── OpenRouter
└── Local
```

Start with 2–3 providers rather than integrating everything immediately.

## Required capabilities

- provider abstraction
- model selection
- structured outputs
- retries
- timeouts
- rate-limit handling
- fallback
- token usage
- cost tracking
- request logging without secrets

## Routing

Start simple:

```text
cheap task → cheap model
complex task → stronger model
```

Later:

```text
task complexity
+
cost
+
latency
+
availability
+
quality
```

## Exit criterion

Changing the LLM provider does not require rewriting the agent.

---

# Phase 5 — Semantic Dataset Understanding

## Goal

Use AI where semantic reasoning provides value.

## Build

LLM receives structured metadata.

It should infer:

- likely column meaning
- semantic relationships
- date relationships
- candidate transformations
- possible feature opportunities
- human-readable warnings

Example:

```text
signup_date
last_purchase

Potential relationship:
customer_lifetime
days_since_last_purchase
```

## Guardrail

LLM proposes.

Python verifies.

## Exit criterion

Agent produces structured, valid hypotheses rather than free-form suggestions.

---

# Phase 6 — Feature Hypothesis Engine

## Goal

Turn semantic understanding into measurable experiments.

## Build

Hypothesis schema:

```json
{
  "name": "...",
  "formula": "...",
  "reason": "...",
  "risk": "...",
  "required_columns": [],
  "availability_assumption": "..."
}
```

## Example

```text
Hypothesis:
Customer inactivity is predictive of churn.

Feature:
days_since_last_purchase

Expected effect:
Increase recall.

Risk:
Prediction timestamp must be available.
```

## Exit criterion

Every feature proposed by the agent can be converted into a deterministic experiment specification.

---

# Phase 7 — Experiment Engine

## Goal

Build the heart of MLPilot.

## Experiment lifecycle

```text
Created
  ↓
Validated
  ↓
Queued
  ↓
Running
  ↓
Completed
  ↓
Evaluated
  ↓
Decision
```

## Every experiment records

- hypothesis
- parent experiment
- dataset version
- feature set
- preprocessing
- model
- parameters
- validation
- metrics
- runtime
- resource usage
- AI cost
- artifacts
- decision
- decision reason

## Exit criterion

An experiment can be reproduced from its stored configuration.

---

# Phase 8 — Agentic Decision Loop

## Goal

Move from "AI suggests features" to genuine agentic experimentation.

## Loop

```text
Experiment history
      ↓
Analyze evidence
      ↓
Identify uncertainty
      ↓
Generate candidates
      ↓
Rank candidates
      ↓
Choose next experiment
      ↓
Run
      ↓
Measure
      ↓
Update history
      ↓
Repeat
```

## Agent output

```json
{
  "next_action": "...",
  "reason": "...",
  "expected_information_gain": 0.71,
  "estimated_cost": 0.02,
  "risk": "low"
}
```

## Important

The agent should NOT simply say:

> "Try X."

It should say:

> "Try X because evidence A/B suggests Y, and this experiment distinguishes between hypotheses H1 and H2."

That is the difference between a chatbot and an experimentation agent.

## Exit criterion

The agent can complete a multi-step experiment sequence without repeating known failed experiments.

---

# Phase 9 — Experiment Graph & Decision Provenance

## Goal

Make the project's strongest visual differentiator.

## Build

Interactive graph:

```text
Baseline
├── Feature A
│   └── CatBoost
│       └── Tuning
├── Feature B [Rejected]
└── Model B [Rejected]
```

Each node shows:

- hypothesis
- experiment
- metric delta
- decision
- reason
- cost

## Exit criterion

A user can trace the final model back to every major decision.

---

# Phase 10 — MVP

## Goal

Create the first portfolio-grade product.

## MVP features

- CSV upload
- target selection
- classification/regression
- profiling
- leakage warnings
- preprocessing
- baseline models
- LLM semantic analysis
- feature hypotheses
- feature experiments
- model comparison
- experiment history
- basic experiment graph
- next-experiment recommendation
- final model
- model card
- training script export

## UX

User experience:

```text
Upload CSV
   ↓
Select target
   ↓
Set objective
   ↓
Analyze
   ↓
Run baseline
   ↓
Agent proposes experiments
   ↓
Run experiments
   ↓
Review graph
   ↓
Select final model
   ↓
Export
```

## MVP demo

Use a real dataset with deliberately inserted:

- missing values
- imbalance
- leakage feature
- weak feature
- useful feature

The demo should visibly show the system catching the leakage and discovering useful features.

## MVP success criterion

The demo is compelling without requiring explanation from you.

---

# Phase 11 — V1 Portfolio / Public Beta

## Goal

Turn MVP into something users can actually try.

## Add

### Frontend

- polished dashboard
- project page
- data explorer
- experiment graph
- agent panel
- model page
- reports

### AI

- 2–4 providers
- BYOK
- cost tracking
- fallback

### ML

- stronger model families
- CatBoost
- XGBoost
- LightGBM
- Optuna
- better validation selection

### Exports

- training script
- prediction script
- model card
- Dockerfile
- FastAPI service

## Exit criterion

A new user can use the product without you walking them through it.

---

# Phase 12 — Deployment Layer

## Goal

Turn a model artifact into a usable service.

## Build

### Hosted inference

```text
POST /predict
```

### Batch prediction

```text
POST /batch-predict
```

### Docker export

### API documentation

### Model versioning

## Security

- isolated inference
- authentication
- rate limits
- request logging
- versioned models

## Exit criterion

A user can go from dataset to working prediction endpoint.

---

# Phase 13 — Monitoring & Feedback

## Goal

Close the loop after deployment.

## Monitor

- latency
- errors
- prediction volume
- feature drift
- data drift
- class distribution
- target performance where labels arrive

## Trigger

```text
Drift detected
   ↓
Agent investigates
   ↓
Generates hypotheses
   ↓
Runs offline experiments
   ↓
Candidate model
   ↓
Human approval
```

## Exit criterion

Production signals can feed back into experimentation.

---

# Phase 14 — Autonomous Experiment Mode

## Goal

Make the product meaningfully agentic.

## Modes

### Suggest

No automatic execution.

### Assisted

Safe experiments auto-run.

### Autonomous

Runs within:

- time budget
- compute budget
- experiment budget
- API budget
- allowed models
- allowed feature operations

## Stop conditions

- improvement threshold
- objective achieved
- budget reached
- diminishing returns
- instability

## Exit criterion

The agent can run a bounded experiment campaign safely.

---

# Phase 15 — Experiment Intelligence

## Goal

Build the deeper product differentiation.

Introduce:

## Experiment utility score

```text
Expected improvement
+
Information gain
+
Uncertainty reduction
-
Compute cost
-
API cost
-
Redundancy
-
Risk
```

The agent ranks experiments by expected value.

This becomes a major differentiator.

---

# Phase 16 — Agent Evaluation

## Goal

Measure whether the agent is actually useful.

Track:

- useful experiment rate
- redundant experiment rate
- invalid experiment rate
- average improvement
- improvement per dollar
- experiments to target
- leakage violations
- premature stopping
- unnecessary compute

Build an internal benchmark.

Example:

```text
Dataset A
Expert next experiment: X
Agent next experiment: X

Dataset B
Expert: Y
Agent: Z

Agent usefulness: ...
```

Do not optimize the agent solely against final model score. A good agent should also reduce wasted experimentation.

---

# Phase 17 — Collaboration

## Goal

Enable team use.

## Add

- organizations
- projects
- members
- roles
- comments
- experiment approvals
- shared provider keys
- shared datasets
- audit trail

Roles:

```text
Owner
Admin
ML Engineer
Developer
Viewer
```

---

# Phase 18 — SaaS Infrastructure

## Goal

Make the product sellable.

## Add

### Authentication

- Google
- GitHub
- email

### Billing

Plans based on:

- experiments
- compute
- storage
- AI usage
- deployments

### Usage dashboard

```text
Experiments
18 / 50

AI usage
$0.12 / $2.00

Storage
1.4 GB / 5 GB
```

### Limits

Hard limits and soft warnings.

---

# Phase 19 — Monetization

## Product route

### Free

- limited projects
- BYOK
- limited experiments
- public/demo datasets
- local exports

### Pro

- more experiments
- hosted AI
- autonomous mode
- deployment
- monitoring
- advanced reports

### Team

- collaboration
- RBAC
- shared projects
- audit logs
- shared provider configuration

### Enterprise

- private deployment
- SSO
- custom retention
- VPC/on-prem
- dedicated compute
- custom integrations

Exact pricing should be tested with users.

---

# Phase 20 — Service Business

This phase can happen much earlier than SaaS scale.

## Offer

> "Give us your dataset and ML objective. We use MLPilot to investigate, experiment, validate, and deliver a reproducible model."

Deliverables:

- data assessment
- leakage report
- experiment report
- final model
- training pipeline
- model card
- prediction API
- monitoring

## Why this matters

The product can generate revenue before the SaaS business is mature.

MLPilot becomes:

```text
Internal ML automation engine
+
Client delivery platform
```

---

# Phase 21 — Domain Playbooks

## Goal

Increase useful experiment quality.

Create playbooks for:

### SaaS

- churn
- retention
- LTV

### Marketing

- lead scoring
- conversion

### Finance

- fraud
- risk

### Operations

- demand
- forecasting

### Cybersecurity

- anomaly detection
- classification
- risk scoring

Each playbook defines:

- common features
- common leakage patterns
- validation strategies
- model families
- experiment strategies

---

# Phase 22 — Data Connectors

Expand beyond CSV.

Priority:

1. Parquet
2. PostgreSQL
3. MySQL
4. object storage
5. Excel
6. APIs
7. warehouses

The connector layer should be modular.

---

# Phase 23 — Advanced ML

Only after tabular workflows are strong.

Add:

- time-series forecasting
- ranking
- anomaly detection
- NLP
- multimodal data
- image features

Do not add all of these at once.

---

# Phase 24 — Continuous ML

Long-term:

```text
Production
   ↓
Monitoring
   ↓
Drift
   ↓
Investigation
   ↓
Experiment campaign
   ↓
Candidate model
   ↓
Evaluation
   ↓
Approval
   ↓
Deployment
```

At this point MLPilot becomes an ML lifecycle platform rather than a model-generation tool.

---

# Phase 25 — Enterprise / Private Deployment

## Add

- SSO
- SCIM
- RBAC
- audit logs
- private networking
- self-hosted deployment
- customer-managed keys
- data retention policies
- custom model providers
- isolated compute
- compliance tooling

---

# Phase 26 — Long-Term Moat

Potential moat areas:

## 1. Experiment corpus

With explicit customer consent, learn from historical experiments.

## 2. Experiment recommendation model

Predict which experiment is likely to produce useful information.

## 3. Domain intelligence

Specialized experiment playbooks.

## 4. Benchmark

Build a benchmark for agentic ML experimentation.

## 5. Cost/performance intelligence

Learn:

> "When is this expensive experiment worth running?"

## 6. Decision provenance

Make every important model decision traceable.

---

# Recommended Build Order

Do NOT build every phase sequentially as if each must be completed before the next.

The practical order should be:

```text
Phase 0
  ↓
Phase 1
  ↓
Phase 2
  ↓
Phase 3
  ↓
Phase 4
  ↓
Phase 5
  ↓
Phase 6
  ↓
Phase 7
  ↓
Phase 8
  ↓
Phase 9
  ↓
Phase 10 MVP
  ↓
Phase 11 Public Beta
  ↓
Phase 12 Deployment
  ↓
Phase 13 Monitoring
  ↓
Phase 14 Autonomous Mode
  ↓
Phase 15 Experiment Intelligence
  ↓
Phase 16 Agent Evaluation
  ↓
Phase 17 Collaboration
  ↓
Phase 18 SaaS
```

Then branch:

```text
                    ┌── SaaS
                    │
Core Product ───────┤
                    │
                    └── ML Service
```

Both can coexist.

---

# First Practical Milestones

## Milestone A — "It works"

Dataset → baseline.

## Milestone B — "It understands"

Dataset → semantic analysis.

## Milestone C — "It experiments"

Hypothesis → experiment → evidence.

## Milestone D — "It thinks"

Evidence → next experiment.

## Milestone E — "It remembers"

Experiment graph.

## Milestone F — "It ships"

Model → reproducible API.

## Milestone G — "It operates"

API → monitoring → retraining proposal.

## Milestone H — "It sells"

Auth → billing → teams → deployment.

---

# Definition of Done for the Core Product

MLPilot's core is complete when a user can:

1. Upload a dataset.
2. Define a prediction objective.
3. Receive a deterministic data assessment.
4. See leakage/validation warnings.
5. Establish a baseline.
6. Receive meaningful feature/model hypotheses.
7. Run controlled experiments.
8. See evidence for each experiment.
9. Understand why experiments were kept/rejected.
10. See the complete experiment graph.
11. Allow the agent to choose the next experiment.
12. Stop the agent within defined budgets.
13. Select a final model.
14. Understand model limitations.
15. Export reproducible training code.
16. Export a model/API.
17. Deploy it.
18. Monitor it.
19. Feed production evidence back into future experiments.

---

# The Product Test

At every stage, ask:

> **If I removed the LLM, would this still be a valid ML system?**

If no, too much is delegated to AI.

Then ask:

> **If I removed the agentic decision layer, would this just be ordinary AutoML/MLflow functionality?**

If yes, the agentic experimentation layer is where the product differentiation must become stronger.

The final product should pass both tests:

```text
Strong deterministic ML foundation
              +
Strong agentic experimentation intelligence
              =
MLPilot
```
