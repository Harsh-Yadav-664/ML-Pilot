"""Remember what the feature proposer did (#56): one Feature row per proposal, accepted or not.

A rejected proposal is kept with the stage that stopped it and the reasons, so the report can
show what was tried. Nothing here decides whether a feature helps: a proposal that passed every
check is stored as ``proposed``, and only the validated gain (#58) can accept it.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Feature, Run
from ml.experiments.acceptance import GainResult
from ml.features.engine import Budget, BudgetedRun
from ml.features.ir import FeatureIR, describe
from ml.features.llm_sql import ProposalRecord

# the proposer's last answer is the feature; a proposal that never parsed has no name of its own
UNNAMED = "unnamed_proposal"


def feature_description(f: Feature) -> str | None:
    """The feature in one English sentence: from its spec, else the rationale it came with."""
    if f.ir:
        with contextlib.suppress(ValueError):  # a spec the schema no longer reads: the rationale
            return describe(FeatureIR.model_validate(f.ir))
    return f.rationale


def _name(record: ProposalRecord, position: int) -> str:
    return record.proposal.name if record.proposal else f"{UNNAMED}_{position}"


def feature_row(
    run_id: str,
    record: ProposalRecord,
    position: int,
    gain: GainResult | None = None,
    vetoed: bool = False,
    rollout: int = 0,
) -> Feature:
    """A Feature row for one proposal. With ``gain`` the proposal reached the gain test: its
    status is then ``accepted`` or ``rejected_gain`` and the paired scores are stored. A
    proposal the person vetoed has the status ``vetoed`` and no gain."""
    p = record.proposal
    guard: dict[str, Any] = {
        "status": record.status,
        "stage": record.stage,
        "reasons": record.reasons,
        "repaired": record.repaired,
        "attempts": [
            {"stage": a.stage, "reasons": a.reasons, "proposal": a.proposal}
            for a in record.attempts
        ],
        "llm": record.llm,
    }
    return Feature(
        id=str(uuid.uuid4()),
        run_id=run_id,
        name=_name(record, position),
        kind="llm_sql",
        sql=record.sql,
        ir=p.ir.model_dump(mode="json") if p is not None and p.ir is not None else None,
        rationale=p.rationale if p is not None else None,
        status="vetoed" if vetoed else _status(record, gain),
        guard_results=guard,
        gain=gain.to_dict() if gain is not None else None,
        rollout=rollout,
    )


_STORED = {"proposed", "rejected_guard", "rejected_duplicate"}


def _status(record: ProposalRecord, gain: GainResult | None) -> str:
    if gain is not None:
        return "accepted" if gain.accepted else "rejected_gain"
    return record.status if record.status in _STORED else "rejected_guard"


async def record_proposals(
    db: AsyncSession, run_id: str, records: list[ProposalRecord]
) -> list[Feature]:
    """Store every proposal of a run. ``no_llm`` records are not stored: nothing was proposed."""
    rows = [
        feature_row(run_id, r, i) for i, r in enumerate(records, start=1) if r.status != "no_llm"
    ]
    db.add_all(rows)
    await db.flush()
    return rows


async def stop_run_at_budget(
    db: AsyncSession, run: Run, budgeted: BudgetedRun, budget: Budget
) -> None:
    """Record the budget and what was used on the run; a reached budget makes it ``stopped``.

    The champion stays what it was: stopping at a budget never discards the best model so far.
    """
    run.budget = budget.as_dict()
    run.budget_used = {**budgeted.used, "stopped": budgeted.stopped}
    if budgeted.stopped is not None:
        run.status = "stopped"
        run.error = f"stopped: budget ({budgeted.stopped}) reached; the champion is kept"
    await db.flush()
