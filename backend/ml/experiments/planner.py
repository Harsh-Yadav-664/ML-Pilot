"""Agentic Experiment Planner."""

from __future__ import annotations

from typing import Any

from ai.context_builder import ContextBuilder
from ai.context_inputs import dataset_from_profile
from ai.gateway import AIGateway
from ai.router import TaskType
from ml.core.interfaces import ProfileResult


class ExperimentPlanner:
    """Agentic planner that generates ML hypotheses based on data profiles."""

    def __init__(self, ai_gateway: AIGateway, builder: ContextBuilder):
        self.gateway = ai_gateway
        self.builder = builder

    async def generate_next_hypothesis(
        self,
        profile: ProfileResult,
        target_column: str,
        objective: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a single, non-redundant feature engineering hypothesis based on history.

        Args:
            profile: Deterministic data profile from DataProfiler.
            target_column: The target variable to predict.
            objective: User-provided goal (e.g., "Maximize recall for churn").
            history: List of previous experiment results (feature name, formula, F1 score).

        Returns:
            A dictionary matching the hypothesis schema.
        """
        system_prompt = (
            "You are an expert Machine Learning Data Scientist. "
            "Your role is to propose feature engineering hypotheses based on a dataset's metadata. "
            "You must output ONLY valid JSON matching the requested schema."
        )

        prompt = (
            self.builder.prompt("planner.hypothesis", system_prompt)
            .dataset(dataset_from_profile(profile, target_column), title="Dataset profile")
            .text("Target column", target_column)
            .text("Objective", objective)
            .facts("History of previous experiments", history)
            .text(
                "",
                "Propose exactly 1 next distinct feature engineering hypothesis. "
                "It must be non-redundant given the history of what has already been tried and "
                "their outcomes. Explain what new feature to create, the logic/formula, why it "
                "helps, potential leakage risks, and explicitly state why it's non-redundant "
                "given the history.",
            )
            .build()
        )

        # JSON schema for a single hypothesis
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the new feature (e.g. 'days_since_last_purchase')",
                },
                "formula": {
                    "type": "string",
                    "description": "High-level formula or logic (e.g. 'current_date - last_purchase_date')",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this helps the model achieve the objective",
                },
                "non_redundant_reasoning": {
                    "type": "string",
                    "description": "Explicit reason why this is non-redundant given the history",
                },
                "risk": {
                    "type": "string",
                    "description": "Potential leakage or missing data risks",
                },
                "required_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing columns required for this feature",
                },
                "availability_assumption": {
                    "type": "string",
                    "description": "Assumption about data availability at prediction time",
                },
            },
            "required": [
                "name",
                "formula",
                "reason",
                "non_redundant_reasoning",
                "risk",
                "required_columns",
                "availability_assumption",
            ],
            "additionalProperties": False,
        }

        # Ask AI Gateway
        result = await self.gateway.complete_structured_result(
            task_type=TaskType.HYPOTHESIZE, prompt=prompt, schema=schema
        )
        # Which provider answered, and whether it was the offline fallback, travels with the idea.
        return {**result.structured, "llm": result.meta()}
