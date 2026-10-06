"""Agentic Experiment Planner."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ai.gateway import AIGateway
from ai.router import TaskType
from ml.core.interfaces import ProfileResult


class ExperimentPlanner:
    """Agentic planner that generates ML hypotheses based on data profiles."""

    def __init__(self, ai_gateway: AIGateway):
        self.gateway = ai_gateway

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
        # Convert ProfileResult to JSON-friendly string, excluding massive lists if any
        profile_dict = asdict(profile)
        # Simplify column_stats to just names and types to save context window and avoid overwhelm
        simplified_stats = {
            col: {"type": stats.get("type", "unknown")}
            for col, stats in profile_dict.get("column_stats", {}).items()
        }
        profile_summary = {
            "rows": profile_dict.get("rows"),
            "columns": profile_dict.get("columns"),
            "missing_rate": profile_dict.get("missing_rate"),
            "target_balance": profile_dict.get("target_balance"),
            "warnings": profile_dict.get("warnings"),
            "columns_info": simplified_stats,
        }

        system_prompt = (
            "You are an expert Machine Learning Data Scientist. "
            "Your role is to propose feature engineering hypotheses based on a dataset's metadata. "
            "You must output ONLY valid JSON matching the requested schema."
        )

        prompt = f"""
Given the following dataset profile:
{json.dumps(profile_summary, indent=2)}

Target Column: {target_column}
Objective: {objective}

History of previous experiments:
{json.dumps(history, indent=2)}

Propose exactly 1 next distinct feature engineering hypothesis. 
It must be non-redundant given the history of what has already been tried and their outcomes.
Explain what new feature to create, the logic/formula, why it helps, potential leakage risks, and explicitly state why it's non-redundant given the history.
        """

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
            task_type=TaskType.HYPOTHESIZE,
            prompt=prompt,
            schema=schema,
            system=system_prompt,
        )
        # Which provider answered, and whether it was the offline fallback, travels with the idea.
        return {**result.structured, "llm": result.meta()}
