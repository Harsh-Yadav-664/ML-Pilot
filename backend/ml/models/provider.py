"""ModelProvider ABC implementation using joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.core.interfaces import ModelProvider


class JobLibModelProvider(ModelProvider):
    """Model provider using joblib for serialization."""

    def train(self, X_train: Any, y_train: Any, params: dict[str, Any]) -> Any:
        """Train a model. Phase 0 stub — dispatches to Phase 1 training code.

        TODO (Phase 1): Dispatch based on params['model_name'] to sklearn/XGB/LGBM.
        """
        raise NotImplementedError("JobLibModelProvider.train() is a Phase 1 feature.")

    def predict(self, model: Any, X: Any) -> Any:
        """Generate predictions."""
        return model.predict(X)

    def save(self, model: Any, path: str) -> str:
        """Serialize model using joblib."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return str(Path(path).resolve())

    def load(self, path: str) -> Any:
        """Deserialize model from path."""
        import joblib

        return joblib.load(path)

    def get_feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Return feature importances if available."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return dict(zip(feature_names, [float(v) for v in importances]))
        if hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = abs(coefs).mean(axis=0)
            return dict(zip(feature_names, [float(v) for v in abs(coefs)]))
        return {}
