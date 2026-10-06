"""Hypotheses API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import DBSession
from app.db.models.hypothesis import Hypothesis
from app.schemas.hypothesis import HypothesisCreate, HypothesisRead, HypothesisUpdate

router = APIRouter(prefix="/hypotheses", tags=["hypotheses"])


@router.post("/", response_model=HypothesisRead, status_code=status.HTTP_201_CREATED)
async def create_hypothesis(data: HypothesisCreate, db: DBSession) -> HypothesisRead:
    h = Hypothesis(id=str(uuid.uuid4()), **data.model_dump())
    db.add(h)
    await db.flush()
    return HypothesisRead.model_validate(h)


@router.get("/experiment/{experiment_id}", response_model=list[HypothesisRead])
async def list_hypotheses(experiment_id: str, db: DBSession) -> list[HypothesisRead]:
    result = await db.execute(select(Hypothesis).where(Hypothesis.experiment_id == experiment_id))
    return [HypothesisRead.model_validate(h) for h in result.scalars().all()]


@router.get("/{hypothesis_id}", response_model=HypothesisRead)
async def get_hypothesis(hypothesis_id: str, db: DBSession) -> HypothesisRead:
    result = await db.execute(select(Hypothesis).where(Hypothesis.id == hypothesis_id))
    h = result.scalar_one_or_none()
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return HypothesisRead.model_validate(h)


@router.patch("/{hypothesis_id}", response_model=HypothesisRead)
async def update_hypothesis(
    hypothesis_id: str, data: HypothesisUpdate, db: DBSession
) -> HypothesisRead:
    result = await db.execute(select(Hypothesis).where(Hypothesis.id == hypothesis_id))
    h = result.scalar_one_or_none()
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(h, field, value)
    await db.flush()
    return HypothesisRead.model_validate(h)
