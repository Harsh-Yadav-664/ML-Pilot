"""What a finished run leaves on disk for the export (#61): the champion model and the scores
it gave the validation rows. Written by the run service when the run ends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.reports.explain import shap_summary
from ml.scoring.score import feature_stats

MODEL_FILE = "model.txt"
REFERENCE_FILE = "reference_validation.csv"
CATEGORIES_FILE = "categories.json"
SHAP_FILE = "shap.json"
STATS_FILE = "feature_stats.json"


def artifact_dir(projects_dir: Path, project_id: str, run_id: str) -> Path:
    return projects_dir / project_id / "runs" / run_id


def save_artifacts(
    directory: Path, champion: Any, labels: pd.DataFrame, train: np.ndarray, val: np.ndarray
) -> bool:
    """Write ``model.txt`` and ``reference_validation.csv``; False if the champion has no model."""
    if champion.model is None:
        return False
    directory.mkdir(parents=True, exist_ok=True)
    champion.model.booster_.save_model(str(directory / MODEL_FILE))
    # text attributes are model columns of dtype category; the model needs the same levels again
    categories = {
        str(c): [str(x) for x in champion.frame[c].cat.categories]
        for c in champion.frame.columns
        if isinstance(champion.frame[c].dtype, pd.CategoricalDtype)
    }
    (directory / CATEGORIES_FILE).write_text(json.dumps(categories, indent=2))
    (directory / SHAP_FILE).write_text(
        json.dumps(shap_summary(champion.model, champion.frame, val))
    )
    per_cutoff = labels.groupby("cutoff_time").size().mean()
    stats = {
        "features": feature_stats(champion.frame, train),
        "entities_per_cutoff": float(per_cutoff),
    }
    (directory / STATS_FILE).write_text(json.dumps(stats))
    rows = labels.iloc[val]
    score = np.asarray(champion.model.predict_proba(champion.frame.iloc[val]))[:, 1]
    reference = pd.DataFrame(
        {
            "entity_id": rows["entity_id"].to_numpy(),
            "cutoff_time": pd.to_datetime(rows["cutoff_time"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "label": rows["label"].astype(int).to_numpy(),
            "score": score,
        }
    )
    reference.to_csv(directory / REFERENCE_FILE, index=False)
    return True
