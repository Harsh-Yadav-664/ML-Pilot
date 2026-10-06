"""MLPilot FastAPI application entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import agent, chat, datasets, experiments, jobs, projects
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.migrations import upgrade_to_head
from app.jobs import handlers  # noqa: F401  (registers the job kinds)
from app.jobs.runner import JobWorker

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown."""
    logger.info("Starting up MLPilot backend...")
    await asyncio.to_thread(upgrade_to_head)
    logger.info("Database schema is at the latest migration.")
    worker = JobWorker()
    await worker.start()  # also marks jobs interrupted by the last shutdown as failed
    yield
    logger.info("Shutting down MLPilot backend.")
    await worker.stop()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agentic ML experimentation platform",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local UI development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
API_PREFIX = "/api/v1"
app.include_router(projects.router, prefix=API_PREFIX)
app.include_router(datasets.router, prefix=API_PREFIX)
app.include_router(experiments.router, prefix=API_PREFIX)
app.include_router(agent.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)


# Health
@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": settings.APP_VERSION})


@app.get("/", tags=["root"])
async def root() -> JSONResponse:
    return JSONResponse(
        {"app": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/docs"}
    )
