"""MLPilot FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.db.session import create_all_tables
from app.api.v1 import projects, datasets, experiments, hypotheses, models, reports, ui

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    logger.info("Starting up MLPilot backend...")
    await create_all_tables()
    logger.info("Database tables ready.")
    yield
    logger.info("Shutting down MLPilot backend.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agentic ML experimentation platform",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local UI development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
API_PREFIX = "/api/v1"
app.include_router(projects.router, prefix=API_PREFIX)
app.include_router(datasets.router, prefix=API_PREFIX)
app.include_router(experiments.router, prefix=API_PREFIX)
app.include_router(hypotheses.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(ui.router, prefix=API_PREFIX)


# Health
@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": settings.APP_VERSION})


@app.get("/", tags=["root"])
async def root() -> JSONResponse:
    return JSONResponse({"app": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/docs"})
