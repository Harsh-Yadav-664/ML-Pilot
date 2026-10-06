import logging

from ai.context_builder import ContextBuilder
from ai.context_inputs import dataset_from_profile
from ai.gateway import AIGateway
from ai.router import TaskType
from ml.core.interfaces import ProfileResult

logger = logging.getLogger(__name__)

# Steps understood by ml.data.preparation.dynamic_builder.DynamicPipelineBuilder
ALLOWED_STEPS = frozenset(
    {
        "impute_median",
        "impute_mean",
        "impute_constant",
        "impute_most_frequent",
        "standard_scale",
        "robust_scale",
        "minmax_scale",
        "onehot_encode",
        "target_encode",
        "log1p",
    }
)

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        }
    },
    "required": ["columns"],
}


class CleaningStrategyError(RuntimeError):
    """The LLM could not produce a usable cleaning strategy."""


class DataCleaningAgent:
    def __init__(self, gateway: AIGateway):
        self.gateway = gateway

    async def generate_cleaning_strategy(
        self, profile: ProfileResult, target_column: str, builder: ContextBuilder
    ) -> dict:
        """
        Takes a ProfileResult and asks the LLM to generate an optimal preprocessing JSON strategy.

        Raises CleaningStrategyError if the LLM call fails or returns an invalid strategy.
        """

        system_prompt = (
            "You are an expert Data Scientist specializing in data preparation and feature engineering. "
            "You will receive a dataset profile containing columns, missing values, skewness, and cardinality. "
            "Your task is to output a strictly valid JSON configuration for a ColumnTransformer builder. "
            "For each column (except the target), specify a list of sequential transformation steps. "
            "Available steps:\n"
            "- 'impute_median', 'impute_mean', 'impute_constant', 'impute_most_frequent'\n"
            "- 'standard_scale', 'robust_scale', 'minmax_scale'\n"
            "- 'onehot_encode', 'target_encode'\n"
            "- 'log1p'\n\n"
            "Rules:\n"
            "1. If a column has high cardinality (>10 unique string values), prefer 'target_encode'.\n"
            "2. If a column has heavy skew, prefer 'log1p' then 'robust_scale'.\n"
            "3. If a column is missing >50% of data, it might be better to drop (don't include in columns). \n"
            "4. For numerical columns, always include an imputer before a scaler.\n"
            "5. For categorical columns, always include an imputer before an encoder.\n"
            "Example output format:\n"
            "{\n"
            '  "columns": {\n'
            '    "age": ["impute_median", "standard_scale"],\n'
            '    "income": ["impute_median", "log1p", "robust_scale"],\n'
            '    "city": ["impute_constant", "target_encode"]\n'
            "  }\n"
            "}"
        )

        prompt = (
            builder.prompt("cleaning.strategy", system_prompt)
            .dataset(
                dataset_from_profile(profile, target_column, skip_target=True),
                title="Dataset profile (the target column is left out)",
            )
            .text("Target column", target_column)
            .text("", "Please provide the JSON configuration.")
            .build()
        )

        try:
            config = await self.gateway.complete_structured(
                TaskType.ANALYZE, prompt, schema=STRATEGY_SCHEMA
            )
        except Exception as e:
            logger.error(f"Failed to generate cleaning strategy: {e}")
            raise CleaningStrategyError(f"LLM call failed: {e}") from e

        return _validate_strategy(config, profile, target_column)


def _validate_strategy(config: object, profile: ProfileResult, target_column: str) -> dict:
    """Check the LLM output against the profile and the allowed steps."""
    if not isinstance(config, dict) or not isinstance(config.get("columns"), dict):
        raise CleaningStrategyError("LLM output has no 'columns' object")

    known = set(profile.column_stats) - {target_column}
    columns: dict[str, list[str]] = {}
    for col, steps in config["columns"].items():
        if col not in known:
            raise CleaningStrategyError(f"LLM output names unknown column {col!r}")
        if not isinstance(steps, list) or not all(isinstance(st, str) for st in steps):
            raise CleaningStrategyError(f"Steps for column {col!r} must be a list of strings")
        bad = [st for st in steps if st not in ALLOWED_STEPS]
        if bad:
            raise CleaningStrategyError(f"Unknown steps for column {col!r}: {bad}")
        columns[col] = steps
    return {"columns": columns}
