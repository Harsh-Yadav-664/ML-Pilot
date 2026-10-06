import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


class DynamicPipelineBuilder:
    """Builds a scikit-learn ColumnTransformer dynamically from a JSON configuration."""

    @staticmethod
    def build(config: dict, target_column: str | None = None) -> ColumnTransformer:
        """
        Builds a preprocessing pipeline.
        config format:
        {
            "columns": {
                "age": ["impute_median", "robust_scale"],
                "income": ["impute_median", "log1p", "standard_scale"],
                "city": ["impute_constant", "target_encode"]
            },
            "drop": ["id", "name"]
        }
        """
        transformers = []

        column_configs = config.get("columns", {})

        for col, steps in column_configs.items():
            if col == target_column:
                continue

            pipeline_steps = []
            for step in steps:
                if step == "impute_median":
                    pipeline_steps.append(("impute", SimpleImputer(strategy="median")))
                elif step == "impute_mean":
                    pipeline_steps.append(("impute", SimpleImputer(strategy="mean")))
                elif step == "impute_constant":
                    pipeline_steps.append(
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing"))
                    )
                elif step == "impute_most_frequent":
                    pipeline_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
                elif step == "standard_scale":
                    pipeline_steps.append(("scale", StandardScaler()))
                elif step == "robust_scale":
                    pipeline_steps.append(("scale", RobustScaler()))
                elif step == "minmax_scale":
                    pipeline_steps.append(("scale", MinMaxScaler()))
                elif step == "onehot_encode":
                    pipeline_steps.append(
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    )
                elif step == "target_encode":
                    # Available in sklearn >= 1.3
                    pipeline_steps.append(("encode", TargetEncoder()))
                elif step == "log1p":
                    pipeline_steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))

            if pipeline_steps:
                transformers.append((f"pipe_{col}", Pipeline(steps=pipeline_steps), [col]))

        # Remainder: pass-through or drop
        remainder = "drop"  # strict by default for advanced cleaning

        # If config is empty, fallback to empty transformer
        if not transformers:
            return ColumnTransformer(transformers=[("empty", "drop", [])], remainder="passthrough")

        return ColumnTransformer(transformers=transformers, remainder=remainder)
