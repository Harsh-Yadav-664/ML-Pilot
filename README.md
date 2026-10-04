# MLPilot

**The ML experiment loop — without the chaos.**

---

## The Problem

Data scientists spend most of their time on work that doesn't require expertise: running random model variants, hunting for why a model's score is suspiciously perfect, figuring out whether a feature from last week was actually worth keeping. There's no record of what was tried, why it was tried, or what the results actually showed.

Most ML projects end with a model and no story — just a notebook with 30 cells and no commit history.

---

## What MLPilot Does

MLPilot runs a structured experiment loop on tabular data:

1. **Profile the dataset** — dimensions, types, missingness, class balance, and a six-category leakage scan with a specific reason per flag.
2. **Establish a deterministic baseline** — reproducible, no hidden preprocessing.
3. **Propose one experiment at a time** — the agent reads the full history of what's already been tried and proposes a single, non-redundant next experiment, with an explicit reason it's not repeating prior work.
4. **Execute safely** — the LLM proposes; deterministic Python validates and runs. No LLM-generated code ever executes on the host.
5. **Record everything** — hypothesis, change, results, decision, and rationale are stored against every run.
6. **Recommend what's next** — grounded in the accumulated evidence, not a fresh guess each time.

---

## What Makes This Different

**Sequential agentic reasoning, not brute-force search.**
The experiment agent does not generate N hypotheses up front and test them in parallel. It reads the outcome of each run before proposing the next one. This means recommendations are actually conditioned on evidence, and redundant experiments are explicitly prevented.

**Categorized leakage detection.**
Six distinct categories — target, missingness, train/test contamination, temporal, preprocessing, and aggregate — each flagged separately with its own evidence. Not a single generic "leakage warning."

**The LLM proposes; code validates and executes. Hard boundary.**
No LLM-generated Python or shell commands execute on the host. Formula evaluation uses a safe AST parser restricted to arithmetic over actual DataFrame columns. This is auditable and explainable — which matters if someone asks why a feature was included.

**Reproducible by default.**
Every completed experiment can be reconstructed from its stored configuration. Exported training scripts reproduce the reported result.

**Direct read-only SQL connection.**
Connect directly to a data warehouse via read-only query instead of forcing a CSV export step, matching how real ML pipelines ingest data.

---

## How to Run

From the repo root:

```bash
python start.py
```

On Windows PowerShell (handles emoji encoding):

```powershell
$env:PYTHONIOENCODING="utf-8"; python start.py
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Keys:** Add to `backend/.env`. The system falls back to `StubProvider` (offline mode) if no key is available.

---

## Architecture

```
React / TypeScript UI  (mlpilot-agentic-platform-ui/)
        |
FastAPI backend  (backend/)
        |
   ┌────┴────┐
AI Gateway   ML Engine
   |              |
TaskRouter    Safe Runner
(availability    (AST eval,
 + complexity     sklearn
 tiering)         pipelines)
```

- **`backend/ai/`** — AI gateway with two routing dimensions: availability fallback (Groq → Gemini → NVIDIA NIM → Stub) and complexity tiering (cheap/fast for formatting; strongest available for high-stakes decisions like dropping a column or declaring a model worthless).
- **`backend/ml/`** — experiment executor, leakage detector, data profiler, and preparation pipeline.
- **`docs/`** — architecture, experiment schema, and interface references.

---

## Project Status

Core loop is implemented and working: baseline, sequential hypothesis-driven experiments, leakage detection, and reproducible export. See [`MLPilot — Master Build Checklist.md`](MLPilot%20—%20Master%20Build%20Checklist.md) for what is verified complete and what is still open.
