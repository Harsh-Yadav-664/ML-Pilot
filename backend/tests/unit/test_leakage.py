"""Leakage scanner: no false positives on the telecom sample, and it finds planted leaks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.validation.leakage import Finding, flagged_columns, name_tokens, scan
from tests.fixtures.leakage.suite import CASES, evaluate

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


def test_only_customer_id_is_flagged_on_telecom():
    findings = scan(pd.read_csv(SAMPLE), "Churn")
    assert flagged_columns(findings) == {"customerID"}
    (fid,) = [f for f in findings if f.flagged]
    assert fid.category == "ID" and fid.evidence["unique_ratio"] == 1.0


@pytest.mark.parametrize(
    "name, tokens",
    [
        ("InternetService", ["internet", "service"]),
        ("Dependents", ["dependents"]),
        ("StreamingTV", ["streaming", "tv"]),
        ("customerID", ["customer", "id"]),
        ("signup_date", ["signup", "date"]),
        ("HTTPStatusCode", ["http", "status", "code"]),
    ],
)
def test_names_are_split_into_whole_tokens(name, tokens):
    assert name_tokens(name) == tokens


def test_planted_leak_suite_precision_and_recall():
    result = evaluate()
    print(
        f"\nplanted-leak suite: precision={result['precision']:.2f} recall={result['recall']:.2f} "
        f"over {len(CASES)} datasets (TP={result['tp']} FP={result['fp']} FN={result['fn']})"
    )
    assert len(CASES) >= 10
    assert result["recall"] >= 0.9, result["cases"]
    assert result["precision"] >= 0.8, result["cases"]


def test_findings_carry_category_severity_and_evidence():
    case = next(c for c in CASES if c.name == "target copy")
    findings = scan(case.build(), case.target)
    copy = next(f for f in findings if f.check == "target_copy")
    assert isinstance(copy, Finding)
    assert (copy.column, copy.category, copy.severity) == ("is_active", "L2", "block")
    assert copy.evidence["rows_checked"] > 0 and copy.explanation
    assert set(copy.to_dict()) == {
        "column",
        "check",
        "category",
        "severity",
        "explanation",
        "evidence",
    }


def test_single_feature_check_works_for_text_targets():
    rng = np.random.default_rng(0)
    y = rng.choice(["Yes", "No"], 400)
    df = pd.DataFrame(
        {"noise": rng.normal(size=400), "leak": (y == "Yes") + rng.normal(0, 0.01, 400), "y": y}
    )
    flagged = {f.column: f for f in scan(df, "y") if f.check == "single_feature_predictiveness"}
    assert set(flagged) == {"leak"} and flagged["leak"].severity == "block"


def test_missing_target_column_is_an_error():
    with pytest.raises(ValueError, match="not found"):
        scan(pd.DataFrame({"a": [1, 2]}), "y")
