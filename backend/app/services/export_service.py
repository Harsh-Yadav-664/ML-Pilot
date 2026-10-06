"""Build the standalone training script for an experiment (the export bundle's first piece)."""

from __future__ import annotations

import json
from pathlib import Path

from app.db.models import Experiment
from ml.data.preparation.feature_frame import ONEHOT_MAX_CATEGORIES
from ml.features import safe_eval as safe_eval_module

# Engines the generated training script can rebuild.
EXPORTABLE_ENGINES = {
    "XGBClassifier",
    "LGBMClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "LogisticRegression",
}


class ExportNotAvailable(Exception):
    """The experiment can't be exported (yet); the message says why."""


def training_script(exp: Experiment) -> str:
    """Return a runnable script that rebuilds the experiment's features and model."""
    if exp.model_name not in EXPORTABLE_ENGINES:
        raise ExportNotAvailable(
            f"Export for the {exp.model_name} engine is not available yet; "
            f"supported: {sorted(EXPORTABLE_ENGINES)}."
        )
    target_encoding = exp.parameters.get("target_encoding")
    if not target_encoding:
        raise ExportNotAvailable(
            "Experiment has no recorded target encoding; run it to completion before exporting."
        )
    classes_literal = json.dumps(target_encoding["classes"])
    engineered = list(exp.parameters.get("features") or [])
    if exp.parameters.get("feature_name") and exp.parameters.get("formula"):
        engineered.append(
            {"name": exp.parameters["feature_name"], "formula": exp.parameters["formula"]}
        )
    features_literal = json.dumps(engineered)
    excluded_literal = json.dumps(sorted(exp.parameters.get("excluded_features", {})))
    numeric_text_literal = json.dumps(sorted(exp.parameters.get("numeric_coercion", {})))
    # The script embeds the same safe evaluator the experiment used (no second copy to drift).
    safe_eval_source = Path(safe_eval_module.__file__).read_text()

    script = f"""# MLPilot Auto-Generated Training Script
# Experiment ID: {exp.id}
# Hypothesis: {exp.hypothesis}

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Install dependencies if needed: pip install scikit-learn pandas xgboost lightgbm

import json

# --- Safe formula evaluator, copied from MLPilot (ml/features/safe_eval.py) ---
{safe_eval_source}
# --- end of safe formula evaluator ---

# Engineered features used by this experiment, in the order they were added
FEATURES = json.loads({features_literal!r})

# ID columns excluded from features, and text columns converted to numbers, when the experiment ran
EXCLUDED_FEATURES = json.loads({excluded_literal!r})
NUMERIC_TEXT_COLUMNS = json.loads({numeric_text_literal!r})

def load_and_prepare_data():
    df = pd.read_csv("{exp.dataset_version}")

    # Same clean-up MLPilot applied when the experiment ran
    df = df.drop(columns=EXCLUDED_FEATURES)
    for col in NUMERIC_TEXT_COLUMNS:
        df[col] = pd.to_numeric(df[col].astype("string").str.strip(), errors="coerce").astype("float64")

    # Feature engineering with the same safe evaluator MLPilot used; an invalid formula raises.
    for feat in FEATURES:
        df[feat["name"]] = evaluate_formula(feat["formula"], df)

    y = df["{exp.parameters.get("target_column", "target")}"]
    X = df.drop(columns=["{exp.parameters.get("target_column", "target")}"])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Class labels in encoded order (index = model output); recorded when the experiment ran
CLASSES = json.loads({classes_literal!r})
CLASS_INDEX = {{label: i for i, label in enumerate(CLASSES)}}

def encode_target(y):
    return y.map(CLASS_INDEX).astype(int)

def decode_predictions(codes):
    return [CLASSES[int(c)] for c in codes]

def build_model():
    # Preprocessor
    numeric_features = ["..."] # Automatically detected in runtime
    categorical_features = ["..."]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories={ONEHOT_MAX_CATEGORIES}, sparse_output=False))
            ]), categorical_features)
        ])
        
    # Model
    model_name = "{exp.model_name}"
    params = {exp.parameters.get("best_params", exp.parameters.get("model_params", {}))}
    
    if model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**params)
    elif model_name == 'LGBMClassifier':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**params)
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**params)
    elif model_name == 'GradientBoostingClassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**params)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**params)
        
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    pipeline = build_model()
    
    # This assumes numeric/categorical lists are populated
    # In practice, you'd dynamically select them here:
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    pipeline.steps[0][1].transformers[0] = ('num', pipeline.steps[0][1].transformers[0][1], numeric_features)
    pipeline.steps[0][1].transformers[1] = ('cat', pipeline.steps[0][1].transformers[1][1], categorical_features)
    
    pipeline.fit(X_train, encode_target(y_train))
    print("Training complete! Accuracy:", pipeline.score(X_test, encode_target(y_test)))
    predictions = decode_predictions(pipeline.predict(X_test))
    print("Sample predictions:", predictions[:10])
"""
    return script
