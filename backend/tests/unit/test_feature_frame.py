"""ID columns are excluded and numbers stored as text are converted, with a report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.preparation.feature_frame import (
    coerce_numeric_text,
    find_id_columns,
    prepare_feature_frame,
)


def _frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "row_number": np.arange(n),
            "plan": rng.choice(["basic", "pro"], size=n),
            "monthly": rng.normal(50, 10, size=n),  # unique floats are features, not IDs
            "tenure": rng.integers(0, 12, size=n),
        }
    )


def test_unique_per_row_and_named_ids_are_found():
    df = _frame()
    df["order_id"] = [f"O{i % 48}" for i in range(50)]  # 48 unique of 50, name ends in _id
    found = find_id_columns(df)
    assert set(found) == {"customerID", "row_number", "order_id"}


def test_features_are_not_mistaken_for_ids():
    found = find_id_columns(_frame())
    for col in ("plan", "monthly", "tenure"):
        assert col not in found
    # A name ending in "id" with few distinct values is a feature (e.g. "paid").
    df = pd.DataFrame({"paid": ["yes", "no"] * 25, "store_id": [1, 2, 3, 4, 5] * 10})
    assert find_id_columns(df) == {}


def test_numeric_text_is_converted_and_failures_counted():
    df = pd.DataFrame(
        {
            "TotalCharges": ["10.5", " 20", " ", "30.25"] * 25,  # 25 blanks out of 100
            "city": ["Pune", "Delhi", "12", "Goa"] * 25,
        }
    )
    # 75% parse: below the threshold, so nothing changes.
    out, failures = coerce_numeric_text(df)
    assert failures == {}
    assert not pd.api.types.is_numeric_dtype(out["TotalCharges"])

    df.loc[df["TotalCharges"] == " ", "TotalCharges"] = ["40"] * 24 + [" "]
    out, failures = coerce_numeric_text(df)
    assert failures == {"TotalCharges": 1}
    assert out["TotalCharges"].dtype == "float64"
    assert out["TotalCharges"].isna().sum() == 1
    assert not pd.api.types.is_numeric_dtype(out["city"])


def test_prepare_reports_both_on_the_telecom_sample():
    from pathlib import Path

    df = pd.read_csv(Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv")
    X, report = prepare_feature_frame(df.drop(columns=["Churn"]))
    assert "customerID" not in X.columns
    assert report["excluded_features"] == {"customerID": "id-like: one unique value per row"}
    assert report["numeric_coercion"] == {"TotalCharges": 11}
    assert X["TotalCharges"].dtype == "float64"
