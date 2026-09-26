# MLPilot

Agentic ML experimentation platform — evidence-driven hypothesis testing for ML projects.

## Tech Stack
- **Backend**: Python 3.13, FastAPI, Pydantic v2, SQLAlchemy 2.x (async), SQLite (dev) / PostgreSQL (prod)
- **ML Engine**: scikit-learn, XGBoost, LightGBM, pandas, numpy
- **AI Gateway**: OpenAI-compatible providers (Groq, NVIDIA NIM, OpenRouter, Cerebras, Mistral) + Gemini + StubProvider
- **Frontend**: Vite + React 18 + TypeScript + Tailwind CSS
- **Infra**: Docker, Alembic, Redis (future)

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Architecture

See `docs/architecture.md` for the full architecture diagram.

## Core Loop

**Hypothesis → Experiment → Evidence → Decision → Next Hypothesis**

The LLM interprets evidence and proposes hypotheses. Deterministic Python owns all data processing, training, metrics, and artifacts.

## Phase 0 Deliverables

- ✅ Provider interfaces (9 ABCs)
- ✅ AI Gateway with fallback chain
- ✅ Experiment schema (PRD-compliant)
- ✅ FastAPI backend skeleton
- ✅ React frontend skeleton
- ✅ StubProvider (fully offline, no API keys needed)
