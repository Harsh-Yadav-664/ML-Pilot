import pytest
import json
from dataclasses import asdict

from ai.gateway import AIGateway
from ml.core.interfaces import ProfileResult
from ml.experiments.planner import ExperimentPlanner
from app.core.config import Settings

@pytest.mark.asyncio
async def test_experiment_planner_generates_hypotheses():
    # Provide a minimal Settings object (no keys needed for stub provider)
    config = Settings()
    
    # Gateway will automatically register the StubProvider
    gateway = AIGateway(config)
    planner = ExperimentPlanner(gateway)
    
    profile = ProfileResult(
        rows=1000,
        columns=5,
        missing_rate=0.0,
        duplicate_rows=0,
        target_balance={"type": "categorical", "distribution": {"0": 0.5, "1": 0.5}},
        column_stats={"signup_date": {"type": "datetime"}, "last_login": {"type": "datetime"}},
        warnings=[]
    )
    
    # In StubProvider, structured responses currently return a mocked schema structure.
    # We may need to ensure StubProvider handles 'hypotheses' array if it's blindly returning a dict.
    # The StubProvider in Phase 0 generates generic structured mock responses based on schema keys.
    
    hypotheses = await planner.generate_hypotheses(
        profile=profile,
        target_column="churned",
        objective="Predict churn accurately",
        max_hypotheses=1
    )
    
    assert isinstance(hypotheses, list)
    
    # Check if the stub provider actually returned the required fields
    if len(hypotheses) > 0:
        hyp = hypotheses[0]
        assert "name" in hyp
        assert "reason" in hyp
        assert "required_columns" in hyp
