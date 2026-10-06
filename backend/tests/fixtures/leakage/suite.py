"""Planted-leak suite for the leakage scanner: small datasets with known leaks and clean controls.

Each case builds a table deterministically and names the leaks planted in it.
A row-level leak (duplicated rows) is named ROWS. Run `python -m tests.fixtures.leakage.suite`
from backend/ to print per-case results and overall precision/recall.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.validation.leakage import scan

ROWS = "<rows>"


@dataclass
class Case:
    name: str
    build: Callable[[], pd.DataFrame]
    target: str
    planted: frozenset


def _customers(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A churn-like table with honest, moderately predictive features."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    logit = (
        -0.04 * tenure + 0.015 * monthly + np.where(contract == "Month-to-month", 1.0, -0.8) - 0.3
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "Contract": contract,
            "InternetService": internet,
            "Dependents": rng.choice(["Yes", "No"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "support_calls_count": rng.poisson(1.5 + churn, n),
            "min_balance": np.round(rng.normal(500, 150, n), 2),
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


def _target_copy():
    df = _customers(seed=1)
    return df.assign(is_active=np.where(df["Churn"] == "Yes", 0, 1))


def _post_outcome_reason():
    df = _customers(seed=2)
    rng = np.random.default_rng(2)
    reasons = rng.choice(["price", "service", "moved"], len(df))
    return df.assign(cancellation_reason=np.where(df["Churn"] == "Yes", reasons, None))


def _refund_amount():
    df = _customers(seed=3)
    rng = np.random.default_rng(3)
    amount = np.where(df["Churn"] == "Yes", rng.uniform(5, 80, len(df)), 0.0)
    return df.assign(refund_amount=np.round(amount, 2))


def _id_encodes_label():
    df = _customers(seed=4).sort_values("Churn", kind="stable").reset_index(drop=True)
    df.insert(0, "account_number", np.arange(100000, 100000 + len(df)))
    return df


def _minmax_scaled():
    df = _customers(seed=5)
    m = df["MonthlyCharges"]
    return df.assign(charges_scaled=(m - m.min()) / (m.max() - m.min()))


def _zscored():
    df = _customers(seed=6)
    m = df["min_balance"]
    return df.assign(balance_z=(m - m.mean()) / m.std(ddof=0))


def _duplicated_rows():
    df = _customers(seed=7)
    return pd.concat([df, df.sample(frac=0.3, random_state=7)], ignore_index=True)


def _noisy_label_proxy():
    df = _customers(seed=8)
    rng = np.random.default_rng(8)
    flip = rng.random(len(df)) < 0.03
    proxy = (df["Churn"] == "Yes") ^ flip
    return df.assign(final_status=np.where(proxy, "lost", "kept"))


def _regression_leak():
    rng = np.random.default_rng(9)
    df = _customers(seed=9).drop(columns=["Churn"])
    spend = df["MonthlyCharges"] * df["tenure"] + rng.normal(0, 50, len(df))
    return df.assign(total_spend=spend, spend_incl_tax=spend * 1.18 + rng.normal(0, 5, len(df)))


def _multiclass_leak():
    df = _customers(seed=10)
    tier = pd.cut(
        df["MonthlyCharges"] + np.random.default_rng(10).normal(0, 25, len(df)),
        [-np.inf, 50, 90, np.inf],
        labels=["basic", "plus", "premium"],
    ).astype(str)
    return df.drop(columns=["Churn"]).assign(
        tier=tier, tier_code=tier.map({"basic": 1, "plus": 2, "premium": 3})
    )


def _clean(seed):
    return lambda: _customers(seed=seed)


def _clean_with_strong_signal():
    df = _customers(seed=13)
    rng = np.random.default_rng(13)
    usage = np.where(df["Churn"] == "Yes", rng.normal(40, 15, len(df)), rng.normal(60, 15, len(df)))
    return df.assign(
        monthly_usage_gb=np.round(usage, 1),
        win_probability=np.round(rng.uniform(0.01, 0.99, len(df)), 3),
    )


CASES = [
    Case("target copy", _target_copy, "Churn", frozenset({"is_active"})),
    Case("post-outcome reason", _post_outcome_reason, "Churn", frozenset({"cancellation_reason"})),
    Case("refund after churn", _refund_amount, "Churn", frozenset({"refund_amount"})),
    Case("ID encodes the label", _id_encodes_label, "Churn", frozenset({"account_number"})),
    Case("min-max scaled on all rows", _minmax_scaled, "Churn", frozenset({"charges_scaled"})),
    Case("z-scored on all rows", _zscored, "Churn", frozenset({"balance_z"})),
    Case("duplicated rows", _duplicated_rows, "Churn", frozenset({ROWS})),
    Case("noisy label proxy", _noisy_label_proxy, "Churn", frozenset({"final_status"})),
    Case("regression target copy", _regression_leak, "total_spend", frozenset({"spend_incl_tax"})),
    Case("multiclass target recode", _multiclass_leak, "tier", frozenset({"tier_code"})),
    Case("clean control A", _clean(11), "Churn", frozenset()),
    Case("clean control B", _clean(12), "Churn", frozenset()),
    Case("clean, strong honest signal", _clean_with_strong_signal, "Churn", frozenset()),
]


def flagged(case: Case) -> set[str]:
    findings = scan(case.build(), case.target)
    return {f.column if f.column is not None else ROWS for f in findings if f.flagged}


def evaluate() -> dict:
    rows = []
    tp = fp = fn = 0
    for case in CASES:
        got = flagged(case)
        hit, extra, missed = got & case.planted, got - case.planted, case.planted - got
        tp, fp, fn = tp + len(hit), fp + len(extra), fn + len(missed)
        rows.append(
            {
                "case": case.name,
                "planted": sorted(case.planted),
                "flagged": sorted(got),
                "missed": sorted(missed),
                "false_flags": sorted(extra),
            }
        )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"cases": rows, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def main() -> None:
    result = evaluate()
    for r in result["cases"]:
        status = "ok" if not r["missed"] and not r["false_flags"] else "MISMATCH"
        print(f"{status:8} {r['case']:30} planted={r['planted']} flagged={r['flagged']}")
    print(
        f"\nPlanted-leak suite: {len(result['cases'])} datasets, TP={result['tp']} FP={result['fp']} FN={result['fn']}"
    )
    print(f"precision={result['precision']:.2f} recall={result['recall']:.2f}")


if __name__ == "__main__":
    main()
