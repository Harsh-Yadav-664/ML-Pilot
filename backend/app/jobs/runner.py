"""Durable job runner: jobs survive restarts, report ordered events and can be cancelled.

Self-hosted and dependency-light: the ``jobs`` table is the queue, the ``job_events``
table is the log, and a small pool of asyncio tasks in the API process does the work.

- ``enqueue()`` adds a ``queued`` row (the caller commits) and wakes the workers.
- A worker claims the oldest queued job with a conditional UPDATE, so two workers (or
  two processes sharing the DB) never run the same job.
- While a job runs, a heartbeat task bumps ``heartbeat_at``. A ``running`` job whose
  heartbeat is older than ``JOB_STALE_SECONDS`` was interrupted (restart, crash, a killed
  worker): ``recover_interrupted()`` marks it ``failed: interrupted by restart`` on
  start-up and on every sweep, instead of leaving it stuck in ``running`` (AGENTS.md rule 8).
- Cancelling sets ``cancel_requested``; handlers call ``ctx.check_cancelled()`` between
  units of work, so a job stops within one step and ends ``cancelled``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import describe
from app.db import session as db_session
from app.db.models import Experiment, Job, JobEvent

logger = logging.getLogger(__name__)

FINAL = ("succeeded", "failed", "cancelled")
INTERRUPTED = "interrupted by restart"

Handler = Callable[["JobContext", dict[str, Any]], Awaitable[dict[str, Any] | None]]
HANDLERS: dict[str, Handler] = {}

# Workers in this process wait on these; enqueue() sets them so new work starts at once.
_wakeups: set[asyncio.Event] = set()


def handler(kind: str) -> Callable[[Handler], Handler]:
    """Register the function that runs jobs of ``kind``."""

    def register(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return register


class JobCancelled(Exception):
    """Raised by ``check_cancelled()``; the runner records the job as ``cancelled``."""


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(ts: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; they are UTC."""
    if ts is None or ts.tzinfo is not None:
        return ts
    return ts.replace(tzinfo=UTC)


def sessions() -> Any:
    # Looked up on each call so tests can point it at their own database.
    return db_session.AsyncSessionLocal


async def enqueue(db: AsyncSession, *, project_id: str, kind: str, params: dict[str, Any]) -> Job:
    """Add a queued job. The caller commits; workers are woken on commit via ``notify()``."""
    if kind not in HANDLERS:
        raise ValueError(f"Unknown job kind {kind!r}")
    job = Job(id=str(uuid.uuid4()), project_id=project_id, kind=kind, params=params)
    db.add(job)
    await db.flush()
    return job


def notify() -> None:
    """Wake idle workers in this process (call after committing an enqueued job)."""
    for event in list(_wakeups):
        event.set()


async def request_cancel(db: AsyncSession, job: Job) -> Job:
    """Cancel a queued job at once, or ask a running one to stop at its next step."""
    if job.status == "queued":
        job.status, job.finished_at = "cancelled", _now()
    elif job.status == "running":
        job.cancel_requested = True
    await db.flush()
    return job


class JobContext:
    """What a handler gets: event log, progress and the cancel check for its job."""

    def __init__(self, job_id: str, next_seq: int) -> None:
        self.job_id = job_id
        self._seq = next_seq

    async def emit(self, type_: str, **payload: Any) -> int:
        """Append an event; returns its ``seq`` (1, 2, 3, ... per job)."""
        self._seq += 1
        async with sessions()() as db:
            db.add(JobEvent(job_id=self.job_id, seq=self._seq, type=type_, payload=payload))
            await db.commit()
        return self._seq

    async def step(self, name: str, progress: float | None = None, **payload: Any) -> None:
        """Record the current step (and progress 0..1) on the job and as a ``step`` event."""
        values: dict[str, Any] = {"current_step": name[:200]}
        if progress is not None:
            values["progress"] = max(0.0, min(1.0, progress))
        async with sessions()() as db:
            await db.execute(update(Job).where(Job.id == self.job_id).values(**values))
            await db.commit()
        await self.emit("step", name=name, progress=values.get("progress"), **payload)

    async def check_cancelled(self) -> None:
        """Raise ``JobCancelled`` if someone asked this job to stop."""
        async with sessions()() as db:
            requested = await db.scalar(select(Job.cancel_requested).where(Job.id == self.job_id))
        if requested:
            raise JobCancelled("cancelled by request")


async def _finish(job_id: str, status: str, **values: Any) -> None:
    async with sessions()() as db:
        await db.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=status, finished_at=_now(), heartbeat_at=_now(), **values)
        )
        await db.commit()


async def recover_interrupted(stale_after: float | None = None) -> list[str]:
    """Mark running jobs with a stale heartbeat ``failed: interrupted by restart``.

    Experiments those jobs had started are marked failed with the same reason, so
    nothing is left looking like it is still running.
    """
    stale = timedelta(seconds=settings.JOB_STALE_SECONDS if stale_after is None else stale_after)
    cutoff = _now() - stale
    recovered: list[str] = []
    async with sessions()() as db:
        running = (await db.scalars(select(Job).where(Job.status == "running"))).all()
        for job in running:
            beat = _aware(job.heartbeat_at) or _aware(job.started_at)
            if beat is not None and beat > cutoff:
                continue
            job.status, job.error, job.finished_at = "failed", INTERRUPTED, _now()
            recovered.append(job.id)
            exp_ids = {job.params.get("experiment_id")}
            events = await db.scalars(select(JobEvent.payload).where(JobEvent.job_id == job.id))
            exp_ids |= {p.get("experiment_id") for p in events}
            exp_ids.discard(None)
            if exp_ids:
                await db.execute(
                    update(Experiment)
                    .where(
                        Experiment.id.in_(exp_ids), Experiment.status.in_(("created", "running"))
                    )
                    .values(status="failed", decision_reason=INTERRUPTED)
                )
        await db.commit()
    for job_id in recovered:
        logger.warning("Job %s was interrupted by a restart; marked failed", job_id)
    return recovered


async def _claim() -> Job | None:
    """Take the oldest queued job, or None. Safe against other workers doing the same."""
    async with sessions()() as db:
        candidates = (
            await db.scalars(
                select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(5)
            )
        ).all()
        for job_id in candidates:
            now = _now()
            claimed = await db.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "queued")
                .values(status="running", started_at=now, heartbeat_at=now)
            )
            await db.commit()
            if claimed.rowcount == 1:  # type: ignore[attr-defined]
                return await db.get(Job, job_id)
    return None


class JobWorker:
    """A few asyncio tasks that claim and run queued jobs, plus a stale-job sweep."""

    def __init__(
        self,
        concurrency: int | None = None,
        heartbeat: float | None = None,
        stale_after: float | None = None,
        idle_poll: float = 1.0,
    ) -> None:
        self.concurrency = concurrency or settings.JOB_WORKERS
        self.heartbeat = heartbeat or settings.JOB_HEARTBEAT_SECONDS
        self.stale_after = settings.JOB_STALE_SECONDS if stale_after is None else stale_after
        self.idle_poll = idle_poll
        self._tasks: list[asyncio.Task[None]] = []
        self._wakeup = asyncio.Event()

    async def start(self) -> None:
        await recover_interrupted(self.stale_after)
        _wakeups.add(self._wakeup)
        self._tasks = [
            asyncio.create_task(self._loop(), name=f"job-worker-{i}")
            for i in range(self.concurrency)
        ]
        self._tasks.append(asyncio.create_task(self._sweep(), name="job-sweeper"))

    async def stop(self) -> None:
        """Stop at once. Running jobs keep status ``running`` and are recovered on start-up."""
        _wakeups.discard(self._wakeup)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _loop(self) -> None:
        while True:
            job = await _claim()
            if job is None:
                self._wakeup.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), self.idle_poll)
                continue
            await self._run(job)

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(max(self.stale_after, 1.0))
            try:
                await recover_interrupted(self.stale_after)
            except Exception:  # the sweep retries next round; the failure is logged loudly
                logger.exception("Stale-job sweep failed")

    async def _beat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat)
            async with sessions()() as db:
                await db.execute(update(Job).where(Job.id == job_id).values(heartbeat_at=_now()))
                await db.commit()

    async def _run(self, job: Job) -> None:
        async with sessions()() as db:
            last = await db.scalar(select(func.max(JobEvent.seq)).where(JobEvent.job_id == job.id))
        ctx = JobContext(job.id, last or 0)
        beat = asyncio.create_task(self._beat(job.id))
        try:
            fn = HANDLERS.get(job.kind)
            if fn is None:
                raise LookupError(f"No handler for job kind {job.kind!r}")
            await ctx.check_cancelled()
            result = await fn(ctx, dict(job.params))
            await _finish(job.id, "succeeded", progress=1.0, result=result)
        except JobCancelled:
            await ctx.emit("log", message="Cancelled by request; stopped before the next step.")
            await _finish(job.id, "cancelled")
        except asyncio.CancelledError:
            # The worker itself is being stopped (shutdown, a killed task): leave the row
            # as 'running' so recovery records it as interrupted, and stop.
            raise
        except Exception as e:  # recorded on the job as status 'failed' with the message
            logger.exception("Job %s (%s) failed", job.id, job.kind)
            await _finish(job.id, "failed", error=describe(e))
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
