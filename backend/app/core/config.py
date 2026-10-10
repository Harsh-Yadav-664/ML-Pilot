"""Application configuration via pydantic-settings."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from mlpilot import __version__


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "MLPilot"
    APP_VERSION: str = __version__
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./mlpilot.db"

    # ── Background jobs (app/jobs/runner.py) ─────────────────────────
    JOB_WORKERS: int = 2  # jobs run at the same time in this process
    JOB_HEARTBEAT_SECONDS: float = 5.0
    # A running job whose heartbeat is older than this was interrupted (restart, crash).
    JOB_STALE_SECONDS: float = 30.0

    # ── Network (AGENTS.md: self-hosted, local by default) ────────────
    # The host start.py binds to; anything but loopback is logged as a warning.
    MLPILOT_HOST: str = "127.0.0.1"
    # Browser origins allowed to call the API (the Vite dev server by default).
    # Set MLPILOT_CORS_ORIGINS as a JSON list or comma-separated.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        validation_alias=AliasChoices("MLPILOT_CORS_ORIGINS", "CORS_ORIGINS"),
    )

    # ── AI Providers (all optional) ──────────────────────────────────
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    NVIDIA_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    CEREBRAS_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    # Local models via Ollama (no key; nothing leaves the machine), e.g. http://localhost:11434
    OLLAMA_BASE_URL: str | None = None

    # ── AI defaults ───────────────────────────────────────────────────
    DEFAULT_AI_PROVIDER: str = "stub"
    DEFAULT_AI_MODEL: str = "stub-default"

    # ── Storage ───────────────────────────────────────────────────────
    ARTIFACTS_DIR: str = "./artifacts"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json

            if v.strip().startswith("["):
                return list(json.loads(v))
            return [o.strip() for o in v.split(",") if o.strip()]
        return list(v)


settings = Settings()
