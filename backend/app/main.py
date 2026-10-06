"""MLPilot FastAPI application entry point."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import agent, chat, datasets, experiments, jobs, projects
from app.core.config import settings
from app.core.local_token import ENV_TOKEN
from app.core.logging import get_logger, setup_logging
from app.core.security import api_token, is_loopback, require_token
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
    api_token()  # create ~/.mlpilot/token now, not on the first request
    if not is_loopback(settings.MLPILOT_HOST) and not os.environ.get(ENV_TOKEN):
        logger.warning(
            "MLPILOT_HOST=%s is reachable from other machines and %s is not set: "
            "the API is protected only by the generated local token file.",
            settings.MLPILOT_HOST,
            ENV_TOKEN,
        )
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

# CORS: only the configured UI origins (MLPILOT_CORS_ORIGINS). The token is sent as a
# header, never a cookie, so no credentials mode is needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers: every API route needs the local token (app/core/security.py); /health doesn't.
API_PREFIX = "/api/v1"
AUTH = [Depends(require_token)]
app.include_router(projects.router, prefix=API_PREFIX, dependencies=AUTH)
app.include_router(datasets.router, prefix=API_PREFIX, dependencies=AUTH)
app.include_router(experiments.router, prefix=API_PREFIX, dependencies=AUTH)
app.include_router(agent.router, prefix=API_PREFIX, dependencies=AUTH)
app.include_router(jobs.router, prefix=API_PREFIX, dependencies=AUTH)
app.include_router(chat.router, prefix=API_PREFIX, dependencies=AUTH)


# Health
@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": settings.APP_VERSION})


@app.get("/", tags=["root"])
async def root() -> JSONResponse:
    return JSONResponse(
        {"app": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/docs"}
    )
