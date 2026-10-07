"""Task feasibility checks (#98): each check at its threshold and not below it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ml.data.schema_graph import build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.feasibility import Check, Thresholds, check_feasibility
from ml.tasks.labels import CutoffCount, TableCoverage, run_on_source
from ml.tasks.spec import from_yaml
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_labels import CHURN

SPEC = from_yaml(CHURN)
# CHURN: monthly cutoffs 2023-04-01 .. 2024-10-01 with a 30 day window, val_from 2024-04-01,
# test_from 2024-07-01: 12 training cutoffs, 3 validation, 4 test.
MONTHS = [(2023 + (3 + i) // 12, (3 + i) % 12 + 1) for i in range(19)]
TRAIN, VAL, TEST = range(12), range(12, 15), range(15, 19)


def counts(
    eligible: list[int] | int = 1000, positives: list[int] | None = None
) -> list[CutoffCount]:
    n = len(MONTHS)
    el = eligible if isinstance(eligible, list) else [eligible] * n
    pos = positives if positives is not None else [el[i] // 10 for i in range(n)]
    out = []
    for i, (y, m) in enumerate(MONTHS):
        cutoff = datetime(y, m, 1, tzinfo=UTC)
        out.append(
            CutoffCount(
                cutoff=cutoff,
                window_end=cutoff.replace(day=1) + (datetime(y, m, 28, tzinfo=UTC) - cutoff),
                eligible=el[i],
                positives=pos[i],
                base_rate=pos[i] / el[i] if el[i] else None,
                mean_label=None,
            )
        )
    return out


def with_positives(train: int, val: int, test: int) -> list[CutoffCount]:
    """Label counts whose split totals are exactly these numbers (spread over each part)."""
    pos = [0] * 19
    for part, total in ((TRAIN, train), (VAL, val), (TEST, test)):
        idx = list(part)
        for k in range(total):
            pos[idx[k % len(idx)]] += 1
    return counts(1000, pos)


def get(report, code: str) -> Check:  # type: ignore[no-untyped-def]
    return next(c for c in report.checks if c.code == code)


def run(c: list[CutoffCount], **kw):  # type: ignore[no-untyped-def]
    return check_feasibility(SPEC, c, **kw)


# -- positives per split ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("train", "val", "test", "status"),
    [
        (49, 400, 400, "block"),  # one below the training minimum
        (50, 400, 400, "warn"),  # at the minimum: not blocked, but below 200
        (400, 19, 400, "block"),
        (400, 20, 400, "warn"),
        (400, 400, 19, "block"),
        (400, 400, 20, "warn"),
        (400, 400, 400, "ok"),
        (200, 200, 200, "ok"),  # at the warning limit
        (200, 199, 200, "warn"),
    ],
)
def test_positives_per_split_at_each_threshold(
    train: int, val: int, test: int, status: str
) -> None:
    check = get(run(with_positives(train, val, test)), "positives_per_split")
    assert check.status == status, check.message
    assert check.numbers == {"train_positives": train, "val_positives": val, "test_positives": test}


def test_a_block_says_which_part_is_short_and_by_how_much() -> None:
    report = run(with_positives(40, 400, 12))
    assert report.status == "block" and report.blocked
    (reason,) = report.reasons
    assert "model: training, test (" in reason  # validation is not short
    assert "40 train" in reason and "12 test" in reason


def test_thresholds_can_be_changed_and_are_stored_in_the_report() -> None:
    t = Thresholds(min_train_positives=10, min_eval_positives=5, warn_positives=20)
    report = run(with_positives(10, 5, 5), thresholds=t)
    assert get(report, "positives_per_split").status == "warn"
    assert report.as_dict()["thresholds"]["min_train_positives"] == 10


# -- base rate per cutoff -----------------------------------------------------------------------


def rates(low: float, high: float) -> list[CutoffCount]:
    pos = [round(1000 * (low if i % 2 else high)) for i in range(19)]
    return counts(1000, pos)


def test_base_rate_ratio_is_fine_at_three_and_a_warning_above() -> None:
    assert get(run(rates(0.05, 0.15)), "base_rate_per_cutoff").status == "ok"  # exactly 3x
    check = get(run(rates(0.05, 0.151)), "base_rate_per_cutoff")
    assert check.status == "warn" and "5.0%" in check.message and "15.1%" in check.message


def test_a_cutoff_without_positives_is_a_warning_with_its_date() -> None:
    pos = [100] * 19
    pos[7] = 0
    check = get(run(counts(1000, pos)), "base_rate_per_cutoff")
    assert check.status == "warn"
    assert check.numbers["cutoffs_without_positives"] == ["2023-11-01"]


# -- eligible entities per cutoff -----------------------------------------------------------------


def eligible_with_drop(drop: float) -> list[CutoffCount]:
    el = [1000] * 19
    for i in range(10, 19):
        el[i] = round(1000 * (1 - drop))
    return counts(el, [e // 10 for e in el])


def test_eligible_drop_is_fine_at_fifty_percent_and_a_warning_above() -> None:
    assert get(run(eligible_with_drop(0.5)), "eligible_per_cutoff").status == "ok"
    check = get(run(eligible_with_drop(0.501)), "eligible_per_cutoff")
    assert check.status == "warn" and "1000 to 499" in check.message
    assert check.numbers["drops"][0]["from"] == "2024-01-01"


def test_a_slow_decline_is_not_a_drop() -> None:
    el = [1000 - 40 * i for i in range(19)]  # 1000 down to 280, never 50% in one step
    assert get(run(counts(el)), "eligible_per_cutoff").status == "ok"


# -- class imbalance and the metric ----------------------------------------------------------------


def test_imbalance_below_one_percent_warns_and_suggests_pr_auc() -> None:
    at_limit = get(run(counts(1000, [10] * 19)), "class_imbalance")
    assert at_limit.status == "ok" and at_limit.numbers["base_rate"] == pytest.approx(0.01)
    # 189 of 19000: 0.995%
    pos = [10] * 19
    pos[0] = 9
    report = run(counts(1000, pos))
    check = get(report, "class_imbalance")
    assert check.status == "warn" and "PR-AUC" in check.message
    assert report.suggested_metric == "pr_auc" and report.metric == "pr_auc"


def test_a_label_that_is_almost_always_one_is_just_as_imbalanced() -> None:
    assert get(run(counts(1000, [995] * 19)), "class_imbalance").status == "warn"


def test_a_different_metric_is_told_to_switch() -> None:
    spec = from_yaml(CHURN.replace("metric: pr_auc", "metric: roc_auc"))
    report = check_feasibility(spec, counts(1000, [5] * 19))
    assert "PR-AUC suits this better than roc_auc" in get(report, "class_imbalance").message
    assert report.metric == "roc_auc" and report.suggested_metric == "pr_auc"


# -- coverage and the horizon ------------------------------------------------------------------------


def test_coverage_warns_only_when_every_related_table_is_nearly_empty() -> None:
    def cov(share: int) -> list[TableCoverage]:
        return [TableCoverage("orders", 1000, share), TableCoverage("refunds", 1000, 0)]

    assert get(run(counts(), coverage=cov(50)), "coverage").status == "ok"  # exactly 5%
    low = get(run(counts(), coverage=cov(49)), "coverage")
    assert low.status == "warn" and "4.9%" in low.message and "orders" in low.message
    assert low.numbers["tables"]["refunds"] == 0.0


def test_no_related_table_is_a_warning_and_not_computed_is_not_a_finding() -> None:
    assert get(run(counts(), coverage=[]), "coverage").status == "warn"
    assert get(run(counts(), coverage=None), "coverage").status == "ok"


def test_dropped_cutoffs_are_reported() -> None:
    assert get(run(counts()), "horizon_vs_data").status == "ok"
    check = get(run(counts(), dropped_cutoffs=2), "horizon_vs_data")
    assert check.status == "warn" and check.numbers == {"cutoffs_kept": 19, "cutoffs_dropped": 2}


# -- the whole report ------------------------------------------------------------------------------------


def test_the_report_status_is_the_worst_check_and_reasons_name_the_blockers() -> None:
    ok = run(counts(1000, [100] * 19))
    assert ok.status == "ok" and ok.reasons == [] and not ok.blocked
    warned = run(counts(1000, [100] * 19), dropped_cutoffs=1)
    assert warned.status == "warn" and not warned.blocked and len(warned.reasons) == 1
    blocked = run(with_positives(1, 1, 1), dropped_cutoffs=1)
    assert blocked.status == "block" and blocked.blocked
    assert all("positives" in r for r in blocked.reasons)  # the warning is not listed as a blocker


def test_a_regression_task_has_no_positives_to_count() -> None:
    spec = from_yaml(
        CHURN.replace("type: binary", "type: regression")
        .replace(', compare: "= 0"', "")
        .replace("metric: pr_auc", "metric: mae")
    )
    c = [CutoffCount(x.cutoff, x.window_end, x.eligible, None, None, 3.0) for x in counts(1000)]
    report = check_feasibility(spec, c)
    assert report.status == "ok"
    assert get(report, "positives_per_split").numbers == {}


# -- coverage read from the demo database -----------------------------------------------------------


def test_coverage_on_the_demo_database_matches_pandas(tmp_path: Path) -> None:
    source = open_source(
        ConnectionSpec(dialect="duckdb", database=str(demo_duckdb(tmp_path / "demo.duckdb")))
    )
    graph = build_schema_graph(source)
    result = run_on_source(SPEC, graph, source, datetime(2025, 1, 1, tzinfo=UTC))

    def table(name: str) -> pd.DataFrame:
        return source.query(f"SELECT * FROM {name}", limit=10**7, timeout_s=300).to_pandas()

    cutoff = pd.Timestamp("2023-04-01")
    customers, orders = table("customers"), table("orders")
    recent = orders[
        (pd.to_datetime(orders.ordered_at).dt.tz_localize(None) >= cutoff - pd.Timedelta(days=90))
        & (pd.to_datetime(orders.ordered_at).dt.tz_localize(None) < cutoff)
    ]
    eligible = set(
        customers[pd.to_datetime(customers.signup_at).dt.tz_localize(None) < cutoff].customer_id
    ) & set(recent.customer_id)
    expected = {}
    for name, column in (
        ("orders", "ordered_at"),
        ("refunds", "refunded_at"),
        ("support_tickets", "opened_at"),
        ("sessions", "started_at"),
        ("marketing_emails", "sent_at"),
    ):
        frame = table(name)
        seen = frame[pd.to_datetime(frame[column]).dt.tz_localize(None) < cutoff]
        expected[name] = len(eligible & set(seen.customer_id))
    got = {c.table: c.covered for c in result.coverage}
    assert all(c.eligible == len(eligible) for c in result.coverage)
    assert {k: got[k] for k in expected} == expected
    assert expected["orders"] == len(eligible)  # eligibility itself needs an order
    print(f"\ncoverage at 2023-04-01 of {len(eligible)} entities: {expected}")
    report = check_feasibility(SPEC, result.counts, coverage=result.coverage)
    assert get(report, "coverage").status == "ok"
