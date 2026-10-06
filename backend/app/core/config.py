"""Application configuration via pydantic-settings."""

from __future__ import annotations

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "MLPilot"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./mlpilot.db"

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

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

            return json.loads(v)
        return v


settings = Settings()
