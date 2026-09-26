"""Classification baseline runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@dataclass
class BaselineResult:
    model_name: str
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
}


class ClassificationBaseline:
    """Run a battery of classification baseline models and return metrics."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = CLASSIFICATION_MODELS

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        return ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ])

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> list[BaselineResult]:
        """Run all baseline classifiers and return a list of BaselineResult."""
        results = []
        preprocessor = self._build_preprocessor(X_train)

        for name, model_cls in self.models.items():
            try:
                # Initialize model, passing random_state if supported
                if 'random_state' in model_cls().get_params():
                    clf = model_cls(random_state=self.random_state)
                else:
                    clf = model_cls()
                    
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
                
                if X_val is None:
                    # Use Cross Validation
                    scoring = ['f1_weighted', 'accuracy']
                    cv_results = cross_validate(pipeline, X_train, y_train, cv=self.cv_folds, scoring=scoring)
                    metrics = {
                        "f1": float(np.mean(cv_results['test_f1_weighted'])),
                        "accuracy": float(np.mean(cv_results['test_accuracy']))
                    }
                else:
                    # Use explicit Validation Set
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_val)
                    from sklearn.metrics import f1_score, accuracy_score
                    metrics = {
                        "f1": float(f1_score(y_val, y_pred, average='weighted', zero_division=0)),
                        "accuracy": float(accuracy_score(y_val, y_pred))
                    }
                
                results.append(BaselineResult(model_name=name, metrics=metrics))
            except Exception as e:
                results.append(BaselineResult(model_name=name, metrics={}, error=str(e)))
        
        return results

    def list_models(self) -> list[str]:
        """Return the list of model names that will be evaluated."""
        return list(self.models.keys())
