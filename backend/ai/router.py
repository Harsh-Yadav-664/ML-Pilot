"""TaskRouter — maps task types to optimal provider/model combinations."""
from __future__ import annotations

from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """ML task types for AI routing."""
    FORMAT = "format"          # cheapest model (formatting output)
    SUMMARIZE = "summarize"    # cheap (summarising data/results)
    HYPOTHESIZE = "hypothesize"  # medium (generating ML hypotheses)
    ANALYZE = "analyze"        # medium (analysing experiment results)
    SYNTHESIZE = "synthesize"  # strong (synthesising multiple experiments)
    REPORT = "report"          # medium (generating reports)
    DECIDE = "decide"          # strongest available (making experiment decisions)


# Provider preference order per task type: list of (provider_name, model_name)
# Earlier entries are preferred. Gateway falls back through this list.
TASK_ROUTING_TABLE: dict[TaskType, list[tuple[str, str]]] = {
    TaskType.FORMAT: [
        ("groq", "qwen/qwen3.8-27b"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.SUMMARIZE: [
        ("groq", "qwen/qwen3.8-27b"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.HYPOTHESIZE: [
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-pro-latest"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.ANALYZE: [
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-pro-latest"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.SYNTHESIZE: [
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-pro-latest"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.REPORT: [
        ("groq", "qwen/qwen3.8-27b"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
    TaskType.DECIDE: [
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-pro-latest"),
        ("gemini", "gemini-flash-latest"),
        ("nvidia_nim", "meta/llama-3.1-nemotron-70b-instruct"),
        ("stub", "stub-default"),
    ],
}


class TaskRouter:
    """Routes AI tasks to the optimal provider/model based on task type and availability."""

    def get_ordered_providers(
        self,
        task_type: TaskType,
        available_providers: list[str],
    ) -> list[tuple[str, str]]:
        """Return an ordered list of (provider, model) pairs for *task_type*.

        Only includes providers that are in *available_providers*.
        Stub is always included last.
        """
        routing = TASK_ROUTING_TABLE.get(task_type, TASK_ROUTING_TABLE[TaskType.SUMMARIZE])
        ordered = [(p, m) for p, m in routing if p in available_providers]
        # Ensure stub is always last
        if "stub" in available_providers and not any(p == "stub" for p, _ in ordered):
            ordered.append(("stub", "stub-default"))
        return ordered

    def get_cheapest_provider(
        self,
        available_providers: list[str],
    ) -> tuple[str, str]:
        """Return the cheapest available provider for simple tasks."""
        # Free providers in preference order
        free_preferences = [
            ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
            ("groq", "llama-3.1-8b-instant"),
            ("stub", "stub-default"),
        ]
        for provider, model in free_preferences:
            if provider in available_providers:
                return provider, model
        return "stub", "stub-default"
