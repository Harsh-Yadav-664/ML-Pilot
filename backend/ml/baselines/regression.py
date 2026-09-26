"""Regression baseline runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RegressionBaselineResult:
    model_name: str
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


REGRESSION_MODELS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    "SVR",
    "KNeighborsRegressor",
]


class RegressionBaseline:
    """Run a battery of regression baseline models.

    Phase 0 stub — run() raises NotImplementedError.
    Full implementation in Phase 1.
    """

    MODELS = REGRESSION_MODELS

    def __init__(self, cv_folds: int = 5, random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state

    def run(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> list[RegressionBaselineResult]:
        """Run all baseline regressors and return results.

        TODO (Phase 1): Implement using sklearn cross_val_score.
        """
        raise NotImplementedError(
            "RegressionBaseline.run() is a Phase 1 feature."
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
