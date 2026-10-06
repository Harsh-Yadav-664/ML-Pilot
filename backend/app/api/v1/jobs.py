"""Background jobs of a project: status, ordered events (poll or stream) and cancel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import DBSession, ProjectID
from app.db.models import Job, JobEvent
from app.jobs import runner
from app.schemas.api import JobEventRead, JobRead

router = APIRouter(prefix="/projects/{project_id}/jobs", tags=["jobs"])

STREAM_POLL_SECONDS = 0.5


async def _job(db: DBSession, project_id: str, job_id: str) -> Job:
    job = await db.get(Job, job_id)
    if job is None or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="Job not found in this project")
    return job


async def _events(db: DBSession, job_id: str, after: int, limit: int) -> list[JobEvent]:
    rows = await db.scalars(
        select(JobEvent)
        .where(JobEvent.job_id == job_id, JobEvent.seq > after)
        .order_by(JobEvent.seq)
        .limit(limit)
    )
    return list(rows.all())


@router.get("/{job_id}", response_model=JobRead)
async def get_job(project_id: ProjectID, job_id: str, db: DBSession) -> JobRead:
    return JobRead.model_validate(await _job(db, project_id, job_id))


@router.get("/{job_id}/events", response_model=list[JobEventRead])
async def get_job_events(
    project_id: ProjectID,
    job_id: str,
    db: DBSession,
    after: int = Query(0, ge=0, description="Return events with seq greater than this"),
    limit: int = Query(500, ge=1, le=1000),
) -> list[JobEventRead]:
    """Events in order of ``seq``. Poll with ``after`` = the last ``seq`` you saw."""
    await _job(db, project_id, job_id)
    return [JobEventRead.model_validate(e) for e in await _events(db, job_id, after, limit)]


@router.get("/{job_id}/events/stream")
async def stream_job_events(
    project_id: ProjectID, job_id: str, db: DBSession, after: int = Query(0, ge=0)
) -> StreamingResponse:
    """The same events as Server-Sent Events, until the job has finished."""
    await _job(db, project_id, job_id)

    async def feed() -> AsyncIterator[str]:
        seen = after
        while True:
            async with runner.sessions()() as session:
                events = await _events(session, job_id, seen, 500)
                status = await session.scalar(select(Job.status).where(Job.id == job_id))
            for e in events:
                seen = e.seq
                body = JobEventRead.model_validate(e).model_dump_json()
                yield f"id: {e.seq}\nevent: {e.type}\ndata: {body}\n\n"
            if status in runner.FINAL and not events:
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                return
            await asyncio.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(feed(), media_type="text/event-stream")


@router.post("/{job_id}/cancel", response_model=JobRead)
async def cancel_job(project_id: ProjectID, job_id: str, db: DBSession) -> JobRead:
    """Cancel a queued job now, or stop a running one at its next step."""
    job = await _job(db, project_id, job_id)
    if job.status in runner.FINAL:
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")
    job = await runner.request_cancel(db, job)
    return JobRead.model_validate(job)
