"""Proposals are stored with the stage and reasons that stopped them (#56)."""

from __future__ import annotations

import uuid

import httpx
from sqlalchemy import select

import app.db.session as db_session
from app.db.models import Feature, Run
from app.services.feature_proposals import record_proposals, stop_run_at_budget
from ml.data.schema_graph import EdgeRef
from ml.features.engine import Budget, BudgetedRun
from ml.features.ir import FeatureIR
from ml.features.llm_sql import Attempt, Proposal, ProposalRecord

IR = FeatureIR(
    name="refund_count_14d",
    entity_table="customers",
    path=[
        EdgeRef(
            from_table="refunds",
            from_columns=("customer_id",),
            to_table="customers",
            to_columns=("customer_id",),
        )
    ],
    source_table="refunds",
    agg="count",
    window_days=14,
)


async def test_every_proposal_is_stored_with_its_stage_and_reasons(
    client: httpx.AsyncClient, project_id: str
) -> None:
    run_id = str(uuid.uuid4())
    good = Proposal(name="refund_count_14d", rationale="Refunds hint at unhappy customers", ir=IR)
    leaky = Proposal(name="orders_soon", rationale="Orders next week", sql="SELECT 1")
    records = [
        ProposalRecord("proposed", None, [], good, "SELECT 1", attempts=[Attempt(None, None, [])]),
        ProposalRecord(
            "rejected_guard",
            "guard",
            ["reads_future: orders.ordered_at is selected from after the cutoff"],
            leaky,
            repaired=True,
            attempts=[Attempt({"name": "orders_soon"}, "guard", ["reads_future"])] * 2,
        ),
        ProposalRecord(
            "rejected_duplicate", "duplicate", ["the same query is already in use"], good
        ),
        ProposalRecord("invalid", "parse", ["answer: not an object"], None),
        ProposalRecord("no_llm", None, ["no LLM answered"]),
    ]
    async with db_session.AsyncSessionLocal() as db:
        db.add(Run(id=run_id, project_id=project_id, task_spec_id=None, engine="not_selected"))
        await db.flush()
        stored = await record_proposals(db, run_id, records)
        await db.commit()
    assert len(stored) == 4  # nothing was proposed by "no_llm"
    async with db_session.AsyncSessionLocal() as db:
        rows = list(
            (
                await db.execute(
                    select(Feature).where(Feature.run_id == run_id).order_by(Feature.name)
                )
            )
            .scalars()
            .all()
        )
    by_name = {r.name: r for r in rows}
    assert {r.kind for r in rows} == {"llm_sql"}
    assert sorted(r.status for r in rows if r.name == "refund_count_14d") == [
        "proposed",
        "rejected_duplicate",
    ]
    leak = by_name["orders_soon"]
    assert leak.status == "rejected_guard" and leak.guard_results is not None
    assert leak.guard_results["stage"] == "guard" and leak.guard_results["repaired"] is True
    assert leak.guard_results["reasons"][0].startswith("reads_future")
    assert len(leak.guard_results["attempts"]) == 2
    duplicate = [r for r in rows if r.status == "rejected_duplicate"]
    assert len(duplicate) == 1 and duplicate[0].ir is not None
    invalid = by_name["unnamed_proposal_4"]
    assert invalid.status == "rejected_guard" and invalid.guard_results["status"] == "invalid"  # type: ignore[index]
    assert invalid.sql is None and invalid.ir is None


async def test_a_run_that_reaches_its_budget_is_stopped_and_says_which(
    client: httpx.AsyncClient, project_id: str
) -> None:
    stopped_id, finished_id = str(uuid.uuid4()), str(uuid.uuid4())
    budget = Budget(max_cost_usd=0.05)
    async with db_session.AsyncSessionLocal() as db:
        for run_id in (stopped_id, finished_id):
            db.add(Run(id=run_id, project_id=project_id, task_spec_id=None, engine="lightgbm"))
        await db.flush()
        hit = BudgetedRun([], "cost", {"cost_usd": 0.06, "seconds": 3.2, "proposals": 3})
        within = BudgetedRun([], None, {"cost_usd": 0.02, "seconds": 1.0, "proposals": 1})
        await stop_run_at_budget(db, await db.get(Run, stopped_id), hit, budget)  # type: ignore[arg-type]
        await stop_run_at_budget(db, await db.get(Run, finished_id), within, budget)  # type: ignore[arg-type]
        await db.commit()
    async with db_session.AsyncSessionLocal() as db:
        stopped = await db.get(Run, stopped_id)
        finished = await db.get(Run, finished_id)
    assert stopped is not None and finished is not None
    assert stopped.status == "stopped" and "budget (cost)" in (stopped.error or "")
    assert stopped.budget == {"max_cost_usd": 0.05, "max_seconds": None, "max_proposals": None}
    assert stopped.budget_used["stopped"] == "cost" and stopped.budget_used["cost_usd"] == 0.06
    assert finished.status == "created" and finished.error is None
    assert finished.budget_used["stopped"] is None
