# MLPilot — Focused Build Phases and Execution Roadmap

**Purpose:** Build a portfolio-grade ML project that can become a useful open-source tool and, if validated, a sellable product or service.

**Guiding principle:** Build the smallest complete experiment loop before building a broad ML platform.

---

## 0. Product Boundary and Working Agreement

### Objective

MLPilot is not being built to compete with the friend's DataClean project. MLPilot's center is hypothesis-led experimentation, evidence, experiment memory, and next-experiment planning.

DataClean's intended center is data cleaning/readiness and a defensible data-to-model workflow. Basic profiling, preprocessing, leakage safeguards, and baseline modeling are necessary supporting capabilities in MLPilot, but should not expand into a parallel DataClean product.

### Rules

- DataClean is optional, not a dependency.
- Build a small native preparation implementation sufficient for MLPilot's workflow.
- Define a stable preparation interface so DataClean can be connected later with agreement on API, license, ownership, privacy, and maintenance.
- Do not copy code or assume access to friend's repository without permission.
- Keep the first product local-first and useful without an LLM.
- Do not implement SaaS, enterprise, deployment, and monitoring before the core experiment loop is validated.

**Exit criterion:** A one-page scope statement, architecture sketch, supported task list, and explicit MVP exclusions are committed to the repository.

---

## Phase 1 — Repository and Product Skeleton

### Goal

Create a maintainable, runnable application without building the entire product at once.

### Tasks

- Inspect the existing Antigravity-generated repository before replacing files.
- Identify what already works, what is mock-only, and what is missing.
- Preserve functioning code unless it conflicts with the architecture or acceptance criteria.
- Establish frontend, backend, ML engine, tests, and artifact directories.
- Add README, setup instructions, environment example, and sample dataset.
- Add formatting, linting, and test commands.

### Suggested structure

```text
mlpilot/
  frontend/
  backend/
    app/
      api/
      projects/
      datasets/
      profiling/
      validation/
      experiments/
      agent/
      providers/
      artifacts/
    tests/
  sample_data/
  docs/
  README.md
  .env.example
```

Adjust to the current repository if a sound structure already exists.

### Exit criteria

- One documented command starts the app locally.
- Frontend can call backend health endpoint.
- Tests run from a clean setup.
- No API key or private dataset is committed.

---

## Phase 2 — Deterministic ML Foundation (No LLM)

### Goal

Prove that MLPilot is a real ML system before adding agentic behavior.

### Build

1. CSV upload and validation.
2. Schema/type inference and dataset fingerprint.
3. Deterministic profiling: shape, missingness, duplicates, cardinality, target distribution, and basic distributions.
4. Target and task selection.
5. Supported validation strategies for binary/multiclass classification and regression.
6. Baseline pipeline using scikit-learn.
7. Correct metrics and fold-level results.
8. Persist experiment configuration and outputs.

Start with a small model set: dummy baseline, logistic/linear model, Random Forest, and one optional gradient-boosting model. Do not integrate every library or AutoML engine at once.

### Critical correctness requirements

- Fit preprocessing within each training fold.
- Keep the test set out of iterative experiment selection when using a sealed holdout.
- Make split strategy, seeds, metric definitions, and package versions inspectable.
- Add tests for target leakage, preprocessing leakage, and reproducibility.
- Report failed checks as failed or unavailable, never as passed.

### Exit criteria

- Given a sample CSV and target, MLPilot produces a reproducible baseline.
- Unit/integration tests verify metrics and split behavior.
- The workflow remains usable with all LLM configuration removed.

---

## Phase 3 — Readiness and Leakage Safeguards

### Goal

Add transparent safeguards that make experiment results more trustworthy.

### Build

- Target/target-derived feature checks.
- Potential post-outcome and unavailable-at-prediction-time warnings.
- Duplicate and entity leakage checks where applicable.
- Basic temporal and group split warnings.
- Suspiciously high metric warnings.
- Clear warning evidence, limitations, and user review actions.

Do not create one opaque “data quality score” that hides the evidence. Do not silently delete features or present heuristics as proof.

### Exit criteria

- Tests include intentionally planted leakage and split-contamination examples.
- UI shows why a warning was raised and what the user can do.
- Known limitations are documented.

---

## Phase 4 — Experiment Specification and Runner

### Goal

Make an experiment a validated, reproducible object—not a prompt that executes arbitrary code.

### Build

Define a versioned experiment schema containing:

- Hypothesis and rationale.
- Parent experiment.
- Dataset fingerprint.
- Exact allowed change specification.
- Model and preprocessing configuration.
- Validation plan and metric.
- Resource/time budget.
- Status, results, artifacts, and decision.

Implement an allowlisted action registry. Initial actions should cover a small number of model comparisons, approved feature transformations, and bounded parameter changes.

### Safety

- Reject unknown actions and invalid configurations.
- Never run LLM-generated Python or shell commands on the host.
- Enforce reasonable file, row, runtime, and experiment limits.
- Record failed/cancelled runs.
- Start with a bounded local worker. Require sandboxed/containerized execution before offering hosted execution of untrusted workloads.

### Exit criteria

- A stored experiment can be rerun from its configuration.
- Invalid or unsupported proposals fail safely and visibly.
- Tests cover run lifecycle and cancellation/failure states.

---

## Phase 5 — Experiment History and Evidence Comparison

### Goal

Build the part that makes MLPilot useful even before sophisticated agent autonomy.

### Build

- Experiment list with status, hypothesis, parent, metric, runtime, and decision.
- Experiment detail with configuration, fold metrics, evidence, and artifacts.
- Parent-versus-candidate comparison.
- Champion selection under the user's metric and constraints.
- Keep/reject/inconclusive decisions with rationale.
- Basic lineage graph or tree with a table fallback.

### Exit criteria

A user can answer: what changed, what happened, compared with what, and why the result was retained or rejected.

---

## Phase 6 — AI Provider and Structured Hypotheses

### Goal

Introduce an LLM only after deterministic profiling, baseline, and experiment execution work.

### Build

- A small AI provider interface.
- Start with one provider and structured output validation.
- Send dataset metadata and carefully selected samples, not entire datasets by default.
- Generate feature/model experiment hypotheses.
- Include rationale, exact proposed action, required columns, prediction-time assumptions, risk, expected evidence, and budget estimate.
- Validate every proposal against the allowlisted experiment schema.

### Requirements

- No LLM is required for profiling, training, metrics, or experiment storage.
- API keys are never logged or committed.
- Handle timeout, rate limit, malformed output, and unavailable provider.
- Show users what information is sent to the provider.
- Offer manual experiment creation if AI is unavailable.

### Exit criteria

- The agent produces structured valid proposals on sample datasets.
- Invalid responses are rejected or retried safely.
- Every proposal is reviewable before execution.

---

## Phase 7 — Evidence-Grounded Interpretation and Next Experiment

### Goal

Deliver MLPilot's defining loop: evidence informs the next experiment.

### Build

After each run, provide the agent with stored experiment results, objective, constraints, prior attempts, and failed/rejected branches.

It should return:

- Summary of measured changes.
- What the evidence supports and does not support.
- Remaining uncertainty.
- Candidate next experiments.
- Why the recommended experiment is informative or non-redundant.
- Risk and cost/runtime estimate.
- Stop recommendation when budgets or useful options are exhausted.

The agent must not fabricate metrics or claim causation from an uncontrolled comparison. Initially use transparent heuristics plus LLM reasoning; do not claim calibrated expected information gain or a learned experiment utility score.

### Exit criteria

- A multi-step experiment sequence can be completed.
- The agent does not repeatedly recommend an already-tried identical action.
- Explanations can be traced to stored evidence.
- User can approve, edit, reject, pause, and stop.

---

## Phase 8 — MVP User Experience and Demo

### Goal

Package the core into a product that a new user can understand and try.

### Build

- Project creation and objective setup.
- Sample dataset onboarding.
- Data overview and warnings.
- Baseline run.
- Agent proposal/approval panel.
- Experiment comparison and history.
- Experiment graph.
- Model selection and export.
- Clear empty, loading, failed, and completed states.

### Demo dataset

Use a public or appropriately licensed tabular dataset. Create a documented demo variant with controlled issues, such as missingness, imbalance, a weak feature, and a deliberately suspicious leakage column. Clearly label synthetic modifications.

### Demo narrative

1. Start a sample project.
2. Show dataset profile and a leakage warning.
3. Run a baseline.
4. Inspect a concrete hypothesis and approve it.
5. Run and compare the experiment.
6. Show the experiment history and why a branch was kept/rejected.
7. Ask for the next experiment.
8. Export the reproducible model artifacts.

### MVP acceptance criteria

- The demo works from a clean environment using documented steps.
- It is compelling without a live explanation.
- No fabricated metrics or fake backend actions are presented as real.
- A user can complete the core loop without DataClean or a paid LLM.

---

## Phase 9 — Optional DataClean Adapter

### Goal

Evaluate integration only after the native interfaces and core workflow exist.

### Preconditions

- Both project owners agree on the integration.
- API contract and ownership are clear.
- License and redistribution rights are understood.
- Data handling and privacy behavior are documented.
- Maintenance responsibility is agreed.

### Tasks

- Implement a thin adapter to the `DataPreparationProvider` interface.
- Keep native preparation available.
- Test outputs and assumptions for consistency.
- Compare correctness, leakage behavior, reproducibility, runtime, coverage, and maintenance.
- Document which provider was used in each project/experiment.

### Exit criteria

Either provider can be selected without rewriting the experiment engine. If integration is not agreed or worthwhile, keep the native provider and continue.

---

## Phase 10 — Public Open-Source/Portfolio Release

### Goal

Make the project independently usable and assess real demand.

### Build

- Clear README with problem, product boundary, architecture, screenshots, demo, setup, limitations, and roadmap.
- License chosen after dependency and contribution review.
- Sample datasets and reproducible demo.
- Issue templates and contribution guidance.
- Local configuration and `.env.example`.
- Basic privacy/security documentation.
- Known limitations and benchmark methodology.
- Export formats and API documentation for stable core interfaces.

### Validation

Ask students, Kaggle participants, developers, and ML practitioners to try a real dataset. Observe where they get stuck and whether they run more than one experiment.

Track:

- First-baseline completion.
- Second-experiment rate.
- Ability to explain a decision.
- Useful/redundant recommendation feedback.
- Repeat use and requested features.

### Exit criteria

At least a small group of external users can install and use the product without direct help. Record feedback honestly; do not infer product-market fit from downloads or stars alone.

---

## Phase 11 — Reliability, Provider Choice, and Stronger ML Support

### Goal

Improve reliability and breadth based on feedback.

Potential additions:

- Optional second/third LLM provider.
- BYOK and provider cost tracking.
- Better model and validation options.
- Improved feature operations and bounded hyperparameter search.
- Optional MLflow export/integration where it reduces duplication.
- More sample datasets and domain playbooks.
- Better test coverage and performance profiling.

Do not add providers or libraries just to inflate the feature list. Every integration must be tested and documented.

---

## Phase 12 — Hosted Product or ML Service (Decision Gate)

This is a decision point, not an automatic phase.

### Option A: Open-source project

Continue improving local workflows, documentation, integrations, reproducibility, and community contribution.

### Option B: Hosted product

Only proceed if users want managed execution or collaboration.

Required before public hosted execution:

- Authentication and authorization.
- Isolated worker containers/sandboxes.
- Resource and usage quotas.
- Secure secret management.
- Tenant and project isolation.
- Object storage and background job queue.
- Dataset retention/deletion controls.
- Rate limits, audit logs, operational monitoring, and incident process.
- Clear terms and privacy policy.

Then consider team workspaces, billing, hosted AI, deployment, and monitoring based on validated demand.

### Option C: ML delivery service

Use MLPilot internally to offer scoped services such as dataset assessment, experiment studies, reproducible model development, model cards, and training pipeline delivery. Keep client data isolated and obtain permission before using it in demos or benchmarks.

### Exit criteria

Select a route based on actual user demand, delivery cost, and willingness to pay—not assumptions made during initial development.

---

## Phase 13 — Long-Term Research and Product Extensions

Only after the core product is reliable and used:

- Better experiment utility and uncertainty-reduction strategies.
- Agent evaluation benchmark across public datasets.
- Domain-specific experiment playbooks.
- Time-series or other task families.
- Collaboration and approvals.
- Hosted inference and monitoring.
- Drift-triggered investigation and retraining proposals.
- Private deployment and enterprise features.

Any claim that MLPilot reduces experiment count or improves outcomes should be supported by a reproducible benchmark against a documented baseline.

---

## Recommended Execution Order

```text
Product boundary + repository audit
             ↓
Deterministic ML foundation
             ↓
Readiness/leakage safeguards
             ↓
Validated experiment runner
             ↓
Experiment history and comparison
             ↓
Structured AI hypotheses
             ↓
Evidence-grounded next-experiment loop
             ↓
Polished MVP and demo
             ↓
Optional DataClean adapter
             ↓
Open-source/public beta
             ↓
User feedback and reliability
             ↓
Decision: open-source, hosted product, or ML service
```

---

## Antigravity Working Instructions

Use this as the execution contract when continuing implementation:

1. Read the current repository and these two documents before making changes.
2. Report what is already implemented, what is mocked, what is broken, and what is missing.
3. Do not rewrite functioning code wholesale without identifying a concrete reason.
4. Work one phase/milestone at a time. Do not implement future SaaS features during the MVP.
5. Before each substantial change, state the files/modules to be changed and the acceptance criteria.
6. Implement real backend behavior. Do not substitute hardcoded metrics, fake progress, or mock agent decisions for working functionality without clearly labeling a prototype.
7. Add tests for ML correctness, leakage prevention, reproducibility, experiment validation, and failure handling.
8. Keep LLM-generated content structured and validate it before use.
9. Never execute arbitrary LLM-generated code or shell commands.
10. Keep DataClean optional and do not copy or depend on its code without explicit agreement.
11. At the end of each milestone, report completed work, tests run and results, known gaps, and the next smallest milestone.
12. Prefer a smaller, correct, demonstrable product over a broad, unreliable feature list.

---

## Definition of Core Completion

The core is complete when a new user can upload or select a dataset, define an objective, inspect deterministic warnings, run a baseline, approve and run a controlled experiment, compare the evidence, inspect the experiment lineage, receive a grounded next-experiment recommendation, and export reproducible artifacts—without requiring DataClean, a paid LLM, or a walkthrough from the author.
