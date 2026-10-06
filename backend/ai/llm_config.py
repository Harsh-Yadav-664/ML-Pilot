"""Which provider and model handles each kind of LLM task, loaded from config.

Tasks are grouped into tiers. Each tier is an ordered list of "provider:model";
the gateway tries them in order and always ends with the offline stub. Keys stay
in environment variables; this file only says who to ask.

    tiers:
      cheap:     [groq:llama-3.1-8b-instant, ollama:llama3.1]
      reasoning: [anthropic:claude-sonnet-4-5, openai:gpt-4o]
      sql:       [ollama:qwen2.5-coder]

The path comes from MLPILOT_LLM_CONFIG (see config/llm.example.yaml). Without it
the built-in DEFAULT_TIERS below are used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ai.router import TaskType

TIERS = ("cheap", "reasoning", "sql")

TASK_TIER: dict[TaskType, str] = {
    TaskType.FORMAT: "cheap",
    TaskType.SUMMARIZE: "cheap",
    TaskType.REPORT: "cheap",
    TaskType.HYPOTHESIZE: "reasoning",
    TaskType.ANALYZE: "reasoning",
    TaskType.SYNTHESIZE: "reasoning",
    TaskType.DECIDE: "reasoning",
    TaskType.SQL: "sql",
}

# Every provider the gateway knows how to register.
KNOWN_PROVIDERS = (
    "groq", "gemini", "nvidia_nim", "openrouter", "cerebras", "mistral", "openai", "anthropic", "ollama", "stub",
)

DEFAULT_TIERS: dict[str, list[str]] = {
    "cheap": [
        "groq:qwen/qwen3.8-27b", "gemini:gemini-flash-latest", "nvidia_nim:meta/llama-3.1-nemotron-70b-instruct",
        "cerebras:llama3.1-8b", "mistral:mistral-small-latest", "openrouter:meta-llama/llama-3.1-8b-instruct:free",
        "openai:gpt-4o-mini", "anthropic:claude-3-haiku-20240307", "ollama:llama3.1",
    ],
    "reasoning": [
        "gemini:gemini-pro-latest", "nvidia_nim:meta/llama-3.1-nemotron-70b-instruct", "groq:openai/gpt-oss-120b",
        "anthropic:claude-3-5-sonnet-20241022", "openai:gpt-4o", "mistral:mistral-large-latest",
        "openrouter:meta-llama/llama-3.1-70b-instruct", "cerebras:llama3.1-70b", "ollama:llama3.1",
    ],
    "sql": [
        "gemini:gemini-pro-latest", "anthropic:claude-3-5-sonnet-20241022", "openai:gpt-4o",
        "groq:openai/gpt-oss-120b", "ollama:qwen2.5-coder",
    ],
}


def parse_entry(entry: str) -> tuple[str, str]:
    """'provider:model' -> (provider, model). The model may itself contain ':'."""
    provider, sep, model = str(entry).partition(":")
    if not sep or not model:
        raise ValueError(f"LLM routing entry {entry!r} must look like 'provider:model'")
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(f"LLM routing entry {entry!r}: unknown provider {provider!r}; known: {', '.join(KNOWN_PROVIDERS)}")
    return provider, model


def validate_tiers(tiers: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(f"Unknown LLM tier(s) {sorted(unknown)}; tiers are {', '.join(TIERS)}")
    out = {}
    for tier in TIERS:
        entries = tiers.get(tier, DEFAULT_TIERS[tier])
        if not isinstance(entries, list):
            raise ValueError(f"LLM tier {tier!r} must be a list of 'provider:model' entries")
        out[tier] = [parse_entry(e) for e in entries]
    return out


def load_routing(path: str | os.PathLike | None = None) -> dict[str, list[tuple[str, str]]]:
    """Routing table per tier. A config file that is missing or malformed raises; no silent default."""
    path = path or os.environ.get("MLPILOT_LLM_CONFIG")
    if not path:
        return validate_tiers(DEFAULT_TIERS)
    import yaml

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"MLPILOT_LLM_CONFIG points to {file}, which does not exist")
    data = yaml.safe_load(file.read_text()) or {}
    if not isinstance(data, dict) or not isinstance(data.get("tiers"), dict):
        raise ValueError(f"{file}: expected a top-level 'tiers:' mapping")
    return validate_tiers(data["tiers"])
