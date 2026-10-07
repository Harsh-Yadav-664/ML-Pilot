"""TaskRouter — maps task types to optimal provider/model combinations."""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """ML task types for AI routing."""

    FORMAT = "format"  # cheapest model (formatting output)
    SUMMARIZE = "summarize"  # cheap (summarising data/results)
    HYPOTHESIZE = "hypothesize"  # medium (generating ML hypotheses)
    ANALYZE = "analyze"  # medium (analysing experiment results)
    SYNTHESIZE = "synthesize"  # strong (synthesising multiple experiments)
    REPORT = "report"  # medium (generating reports)
    DECIDE = "decide"  # strongest available (making experiment decisions)
    SQL = "sql"  # writing SQL features (relational tasks)
    SPEC = "spec"  # drafting a prediction task spec from a question (relational tasks)


class TaskRouter:
    """Routes each task to an ordered list of (provider, model) from the routing config."""

    def __init__(self, routing: dict[str, list[tuple[str, str]]] | None = None) -> None:
        from ai.llm_config import load_routing

        self.routing = routing if routing is not None else load_routing()

    def get_ordered_providers(
        self,
        task_type: TaskType,
        available_providers: list[str],
    ) -> list[tuple[str, str]]:
        """(provider, model) pairs for *task_type*, registered providers only, stub always last."""
        from ai.llm_config import TASK_TIER

        tier = TASK_TIER.get(task_type, "cheap")
        ordered = [
            (p, m) for p, m in self.routing[tier] if p in available_providers and p != "stub"
        ]
        if "stub" in available_providers:
            ordered.append(("stub", "stub-default"))
        return ordered
