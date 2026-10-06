"""Tests for ExperimentPlanner with the offline stub provider."""

from __future__ import annotations

from ai.context_builder import ContextBuilder
from ml.core.interfaces import ProfileResult
from ml.experiments.planner import ExperimentPlanner
from tests.fixtures.gateway import stub_gateway


def make_profile() -> ProfileResult:
    return ProfileResult(
        rows=1000,
        columns=3,
        missing_rate=0.0,
        duplicate_rows=0,
        target_balance={"type": "categorical", "distribution": {"0": 0.5, "1": 0.5}},
        column_stats={"signup_date": {"type": "datetime"}, "last_login": {"type": "datetime"}},
        warnings=[],
    )


async def test_generate_next_hypothesis_returns_schema_fields():
    planner = ExperimentPlanner(stub_gateway(), ContextBuilder())

    hyp = await planner.generate_next_hypothesis(
        profile=make_profile(),
        target_column="churned",
        objective="Predict churn accurately",
        history=[
            {"name": "tenure_days", "formula": "last_login - signup_date", "status": "proposed"}
        ],
    )

    for key in (
        "name",
        "formula",
        "reason",
        "non_redundant_reasoning",
        "risk",
        "required_columns",
        "availability_assumption",
    ):
        assert key in hyp
    assert isinstance(hyp["required_columns"], list)
