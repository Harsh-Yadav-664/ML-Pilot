"""Say in two sentences why a feature was kept or rejected. The decision was already made in code.

The model is given the facts of the rule's decision and asked to explain them; it cannot change
the decision, and if it fails the decision stands with ``explanation_mode: fallback``.
"""

from __future__ import annotations

import logging

from ai.context_builder import ContextBuilder
from ai.gateway import AIGateway
from ai.router import TaskType

logger = logging.getLogger(__name__)


async def explain_decision(
    gateway: AIGateway,
    builder: ContextBuilder,
    *,
    name: str,
    formula: str | None,
    decision: str,
    facts: str,
) -> tuple[str, str]:
    """(explanation, explanation_mode); ("", "fallback") if the model could not answer."""
    try:
        explained = await gateway.complete_result(
            task_type=TaskType.SUMMARIZE,
            prompt=builder.prompt(
                "agent.explain_decision",
                "You explain ML experiment decisions plainly. Use only the facts given.",
            )
            .text(
                "",
                f"A candidate feature '{name}' = {formula} was {decision}ed by a fixed "
                f"statistical rule. Facts: {facts} In two sentences, explain this decision to a "
                "data analyst. Do not change or second-guess the decision.",
            )
            .build(),
        )
    except Exception as e:  # noqa: BLE001 - the rule decision stands; the mode records 'fallback'
        logger.error("Decision explanation failed: %s", e)
        return "", "fallback"
    return explained.text, explained.decision_mode
