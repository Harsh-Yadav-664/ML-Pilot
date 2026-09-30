import json
from ai.gateway import AIGateway
from ml.core.interfaces import ProfileResult
import logging

logger = logging.getLogger(__name__)

class DataCleaningAgent:
    def __init__(self, gateway: AIGateway):
        self.gateway = gateway

    async def generate_cleaning_strategy(self, profile: ProfileResult, target_column: str) -> dict:
        """
        Takes a ProfileResult and asks the LLM to generate an optimal preprocessing JSON strategy.
        """
        
        system_prompt = (
            "You are an expert Data Scientist specializing in data preparation and feature engineering. "
            "You will receive a dataset profile containing columns, missing values, skewness, and cardinality. "
            "Your task is to output a strictly valid JSON configuration for a ColumnTransformer builder. "
            "Do not output markdown code blocks like ```json, just output the raw JSON object. "
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
            "  \"columns\": {\n"
            "    \"age\": [\"impute_median\", \"standard_scale\"],\n"
            "    \"income\": [\"impute_median\", \"log1p\", \"robust_scale\"],\n"
            "    \"city\": [\"impute_constant\", \"target_encode\"]\n"
            "  }\n"
            "}"
        )

        user_prompt = f"""
Dataset Profile:
Total Rows: {profile.rows}
Total Columns: {profile.columns}
Target Column: {target_column}

Column Details:
"""
        for col, stats in profile.column_stats.items():
            if col == target_column:
                continue
            user_prompt += f"- {col}: dtype={stats.get('dtype')}, missing={stats.get('missing_count')} ({stats.get('missing_rate', 0):.1%}), unique={stats.get('unique_count')}, skew={stats.get('skew', 'N/A')}\n"

        user_prompt += "\nPlease provide the JSON configuration."

        try:
            response = await self.gateway.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # clean response if there are markdown tags
            cleaned = response.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
                
            config = json.loads(cleaned)
            return config
            
        except Exception as e:
            logger.error(f"Failed to generate cleaning strategy: {e}")
            # Fallback to empty config if failed
            return {"columns": {}}
