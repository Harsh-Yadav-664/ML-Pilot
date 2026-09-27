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

    async def generate_hypotheses(
        self,
        profile: ProfileResult,
        target_column: str,
        objective: str,
        max_hypotheses: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate a list of feature engineering hypotheses.

        Args:
            profile: Deterministic data profile from DataProfiler.
            target_column: The target variable to predict.
            objective: User-provided goal (e.g., "Maximize recall for churn").
            max_hypotheses: Maximum number of ideas to generate.

        Returns:
            List of dictionaries matching the hypothesis schema.
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

Propose up to {max_hypotheses} distinct feature engineering hypotheses.
Each hypothesis must explain what new feature to create, the logic/formula, why it helps the objective, and potential data leakage risks.
        """

        # JSON schema for the list of hypotheses
        schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the new feature (e.g. 'days_since_last_purchase')"},
                            "formula": {"type": "string", "description": "High-level formula or logic (e.g. 'current_date - last_purchase_date')"},
                            "reason": {"type": "string", "description": "Why this helps the model achieve the objective"},
                            "risk": {"type": "string", "description": "Potential leakage or missing data risks"},
                            "required_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Existing columns required for this feature"
                            },
                            "availability_assumption": {"type": "string", "description": "Assumption about data availability at prediction time"}
                        },
                        "required": ["name", "formula", "reason", "risk", "required_columns", "availability_assumption"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["hypotheses"],
            "additionalProperties": False
        }

        # Ask AI Gateway
        response = await self.gateway.complete_structured(
            task_type=TaskType.HYPOTHESIZE,
            prompt=prompt,
            schema=schema,
            system=system_prompt,
        )

        return response.get("hypotheses", [])
