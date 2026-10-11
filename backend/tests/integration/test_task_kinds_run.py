"""Regression and multi-class tasks run end to end on the demo database (#104).

Each task goes through the API and the job runner with the offline stub provider (a scripted
model proposes features, forced through the gain rule), then its export is verified by the
bundle's own score.py: the validation MAE or log loss recomputed from the database.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from tests.integration.test_export_bundle import connection, exported_run, run_script  # noqa: F401
from tests.integration.test_run_loop_api import runs_url
from tests.unit.test_task_spec import SPEND

STATUS = """
name: next_order_status
description: Whether the next order of a customer is completed, cancelled or returned.
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
  - exists: {table: orders, where: "ordered_at >= :cutoff - interval '90 days' AND ordered_at < :cutoff"}
target:
  type: multiclass
  expression: {table: orders, agg: first, column: status}
horizon: 30d
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
metric: macro_f1
"""

CASES = [
    pytest.param(SPEND, "mae", "trivial_mae", "MAE", id="regression-spend-next-30-days"),
    pytest.param(
        STATUS, "log_loss", "trivial_log_loss", "log loss", id="multiclass-next-order-status"
    ),
]


@pytest.mark.parametrize(("spec", "metric", "trivial", "label"), CASES)
async def test_the_task_runs_is_reported_and_its_export_reproduces_the_metric(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    spec: str,
    metric: str,
    trivial: str,
    label: str,
) -> None:
    root, db_path, run_id = await exported_run(
        client, project_id, connection, monkeypatch, tmp_path, spec
    )
    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "completed" and state["error"] is None, state
    val, test = state["champion_val_metrics"], state["test_metrics"]
    assert test is not None and state["test_error"] is None
    print(
        f"\n{label}: validation {val[metric]:.4f} vs trivial {val[trivial]:.4f}; test {test[metric]:.4f}"
    )
    if metric == "mae":
        # Spend is mostly zero, so the training median (0) is hard to beat on MAE alone; the model
        # is judged by what it explains and by the value in its top 10% (random picking holds 10%)
        assert val["r2"] > 0 and val["captured_value_at_10pct"] > 0.10, val
    else:
        assert val[metric] < val[trivial], val  # beats the training class shares

    report = await client.get(runs_url(project_id, f"/{run_id}/report?format=markdown"))
    assert report.status_code == 200, report.text
    assert label in report.text and "PR-AUC" not in report.text

    env = {**os.environ, "MLPILOT_DB_URL": f"duckdb:///{db_path}"}
    verified = await run_script(root, env, "--verify")
    print(verified.stdout, verified.stderr)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert f"validation {metric} reported" in verified.stdout

    ranked = await client.post(runs_url(project_id, f"/{run_id}/score"), json={})
    assert ranked.status_code == 422 and "not available" in ranked.json()["detail"]
