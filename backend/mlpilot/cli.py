"""The ``mlpilot`` command line (issue #68).

Every command calls the same services as the HTTP API (``app/services``) in this process, with
the same database sessions, SQL guard and job runner; there is no HTTP between the command and
the work. The state (connections, tasks, runs) lives in the same metadata database the web UI
reads, so a run started here shows up in ``mlpilot ui``.

    mlpilot connect --name demo --dialect postgres --host localhost --database demo ...
    mlpilot task draft --question "Which customers will stop ordering in the next 30 days?"
    mlpilot task confirm
    mlpilot run
    mlpilot report

Nothing from ``app``, ``ai`` or ``ml`` is imported at module level: the environment (where the
metadata database and the data live) has to be set before ``app.core.config`` is first read.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import threading
import time
import webbrowser
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import typer

from mlpilot import __version__

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.models import Connection, Run, TaskSpec

app = typer.Typer(
    name="mlpilot",
    help="Ask your database a prediction question. Get an answer you can check.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # locals can hold a connection's password
    rich_markup_mode=None,
)
task_app = typer.Typer(
    help="Draft a prediction task from a question, and confirm it.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode=None,
)
app.add_typer(task_app, name="task")

ConnectionOption = Annotated[
    str | None,
    typer.Option("--connection", "-c", help="Connection name or id (default: the latest)."),
]
TaskOption = Annotated[
    str | None, typer.Option("--task", "-t", help="Task name or id (default: the latest).")
]
RunOption = Annotated[
    str | None,
    typer.Option("--run", "-r", help="Run id or its first characters (default: the latest)."),
]

EXIT_FAILED = 1
EXIT_NEEDS_ANSWER = 3  # `task draft` was asked something it needs the user to say more about
FINAL_RUN = ("completed", "failed", "cancelled", "stopped")
FINAL_JOB = ("succeeded", "failed", "cancelled")


# -- environment ---------------------------------------------------------------------------------


def mlpilot_home() -> Path:
    return Path(os.environ.get("MLPILOT_HOME") or Path.home() / ".mlpilot").expanduser()


def configure_environment() -> None:
    """Keep MLPilot's state in one folder (``~/.mlpilot`` unless ``MLPILOT_HOME`` says otherwise).

    A source checkout keeps its database and data next to the code; an installed package must
    not write into ``site-packages`` or into whatever folder the command is run from. Settings
    are read when ``app`` is first imported, so this only has an effect before that: once the
    app is loaded (a test that already imported it) it leaves everything as it is.
    """
    if "app.core.config" in sys.modules:
        return
    home = mlpilot_home()
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLPILOT_HOME", str(home))
    os.environ.setdefault(
        "MLPILOT_TOKEN_FILE", str(home / "token")
    )  # the default: ~/.mlpilot/token
    os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{home / 'mlpilot.db'}")


def migrate() -> None:
    """Bring the metadata database to the latest schema (what the API does on start-up)."""
    from app.db.migrations import upgrade_to_head

    upgrade_to_head()


def _version(show: bool) -> None:
    if show:
        typer.echo(f"mlpilot {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    configure_environment()


# -- helpers -------------------------------------------------------------------------------------


def die(message: str, code: int = EXIT_FAILED) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def say(message: str = "") -> None:
    typer.echo(message)


def detail_text(detail: Any) -> str:
    """An HTTPException's detail as lines of text (the services raise them for every refusal)."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        lines = [str(detail["message"])] if detail.get("message") else []
        lines += [f"  - {r}" for r in detail.get("reasons", [])]
        for issue in detail.get("issues", []):
            lines.append(
                f"  - {issue.get('path')}: {issue.get('message')} ({issue.get('severity')})"
            )
        if lines:
            return "\n".join(lines)
    return str(detail)


def run_async[T](main_fn: Callable[[], Awaitable[T]]) -> T:
    """Run one command: migrate, run its coroutine, turn a refused request into an exit code."""
    from fastapi import HTTPException

    migrate()

    async def guarded() -> T:
        from app.db import session as db_session

        try:
            return await main_fn()
        finally:
            await db_session._engine.dispose()

    try:
        return asyncio.run(guarded())
    except HTTPException as e:
        die(detail_text(e.detail))


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    """A database session that commits when the block ends, like the API's ``get_db``."""
    from app.db import session as db_session

    async with db_session.AsyncSessionLocal() as db:
        try:
            yield db
            await db.commit()
        except BaseException:
            await db.rollback()
            raise


async def project_id(db: AsyncSession) -> str:
    """The project the web UI would open: the newest one, made on first use."""
    from app.api.deps import DEFAULT_OWNER_ID
    from app.schemas.project import ProjectCreate, TaskType
    from app.services.project_service import ProjectService

    service = ProjectService(db)
    existing, _ = await service.list_by_owner(DEFAULT_OWNER_ID, 1, 1)
    if existing:
        return existing[0].id
    made = await service.create(
        ProjectCreate(name="My first project", task_type=TaskType.BINARY_CLASSIFICATION),
        DEFAULT_OWNER_ID,
    )
    return made.id


def _pick(items: list[Any], ref: str | None, what: str, hint: str) -> Any:
    """The item named by ``ref`` (id, id prefix or name), or the newest when ``ref`` is empty."""
    if not items:
        die(f"There is no {what} yet. {hint}")
    if ref is None:
        return items[-1]
    exact = [i for i in items if ref in (i.id, getattr(i, "name", None))]
    found = exact or [i for i in items if i.id.startswith(ref)]
    if len(found) != 1:
        names = ", ".join(f"{getattr(i, 'name', i.id[:8])} ({i.id[:8]})" for i in items)
        die(f"{ref!r} does not name exactly one {what}. Known: {names}")
    return found[0]


async def pick_connection(db: AsyncSession, pid: str, ref: str | None) -> Connection:
    from app.services.connection_service import ConnectionService

    connections = await ConnectionService(db).list(pid)  # oldest first
    return _pick(connections, ref, "connection", "Add one with `mlpilot connect`.")


async def pick_task(
    db: AsyncSession, pid: str, ref: str | None, status: Literal["draft", "confirmed"] | None
) -> TaskSpec:
    from sqlalchemy import select

    from app.db.models import TaskSpec

    query = select(TaskSpec).where(TaskSpec.project_id == pid).order_by(TaskSpec.created_at)
    if status is not None and ref is None:
        query = query.where(TaskSpec.status == status)
    rows = list((await db.scalars(query)).all())
    hint = (
        "Draft one with `mlpilot task draft --question ...`."
        if status != "confirmed"
        else "Confirm one with `mlpilot task confirm`."
    )
    return _pick(rows, ref, f"{status} task" if status else "task", hint)


async def pick_run(db: AsyncSession, pid: str, ref: str | None) -> Run:
    from sqlalchemy import select

    from app.db.models import Run

    rows = list(
        (await db.scalars(select(Run).where(Run.project_id == pid).order_by(Run.created_at))).all()
    )
    return _pick(rows, ref, "run", "Start one with `mlpilot run`.")


def table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        say("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())


def day(value: datetime | str | None) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d") if isinstance(value, datetime) else str(value)[:10]


def number(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


# -- connect and schema --------------------------------------------------------------------------


@app.command()
def connect(
    name: Annotated[str, typer.Option(help="A name for this connection.")],
    database: Annotated[
        str, typer.Option(help="Postgres: the database name. SQLite and DuckDB: the file path.")
    ],
    dialect: Annotated[str, typer.Option(help="postgres, sqlite or duckdb.")] = "postgres",
    host: Annotated[str | None, typer.Option(help="Postgres only.")] = None,
    port: Annotated[int | None, typer.Option(help="Postgres only (default 5432).")] = None,
    user: Annotated[str | None, typer.Option(help="Postgres only. Use a read-only role.")] = None,
    password_env: Annotated[
        str | None,
        typer.Option(
            help="Name of the environment variable that holds the password. The password is "
            "never taken on the command line, so it stays out of your shell history."
        ),
    ] = None,
    ssl_mode: Annotated[
        str | None,
        typer.Option(help="Postgres only: disable, prefer, require, verify-ca or verify-full."),
    ] = None,
) -> None:
    """Save a database connection and test it (read-only: nothing is ever written to it)."""
    from pydantic import ValidationError

    from app.schemas.connection import ConnectionCreate
    from app.services.connection_service import ConnectionService

    if dialect not in ("postgres", "sqlite", "duckdb"):
        die("--dialect must be postgres, sqlite or duckdb")
    if password_env and password_env not in os.environ:
        die(f"The environment variable {password_env} is not set in this shell")
    try:
        data = ConnectionCreate.model_validate(
            {
                "name": name,
                "dialect": dialect,
                "host": host,
                "port": port,
                "database": database,
                "username": user,
                "password_env": password_env,
                "ssl_mode": ssl_mode,
            }
        )
    except ValidationError as e:
        die("; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()))

    async def go() -> tuple[bool, str | None, bool | None]:
        async with session() as db:
            pid = await project_id(db)
            service = ConnectionService(db)
            conn = await service.create(pid, data)
            result = await service.test(conn)
        if not result.ok:
            say(f"Saved connection {name!r}, but it could not connect: {result.message}")
            return False, result.error_code, None
        say(f"Connected to {name!r}: {result.server_version} ({result.latency_ms} ms).")
        if result.can_write:
            typer.secho(
                "Warning: this role can write to the database. MLPilot only ever reads, but "
                "connect with a read-only role. " + " ".join(result.privilege_notes),
                fg=typer.colors.YELLOW,
                err=True,
            )
        else:
            say("The role is read-only.")
        return True, None, result.can_write

    ok, _, _ = run_async(go)
    if not ok:
        raise typer.Exit(EXIT_FAILED)
    say('Next: `mlpilot schema`, then `mlpilot task draft --question "..."`.')


@app.command()
def schema(connection: ConnectionOption = None) -> None:
    """Show the tables, time columns and relationships MLPilot found in the database."""
    from app.services.connection_service import ConnectionService

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            conn = await pick_connection(db, pid, connection)
            graph = await ConnectionService(db).schema_graph(conn)
        say(f"{conn.name} ({graph.dialect}): {len(graph.tables)} tables")
        table(
            [
                [
                    t.key,
                    f"{t.row_count:,}" + ("~" if t.row_count_estimated else ""),
                    t.time_column or ("static" if t.is_static else "-"),
                    str(len(t.columns)),
                ]
                for t in graph.tables
            ],
            ["table", "rows", "time column", "columns"],
        )
        say()
        say(f"{len(graph.edges)} relationships")
        table(
            [
                [
                    f"{e.from_table}({', '.join(e.from_columns)})",
                    f"{e.to_table}({', '.join(e.to_columns)})",
                    e.source,
                    e.cardinality,
                ]
                for e in graph.edges
            ],
            ["from", "to", "found by", "kind"],
        )
        for warning in graph.warnings:
            say(f"warning: {warning}")

    run_async(go)


# -- tasks ---------------------------------------------------------------------------------------


@task_app.command("draft")
def task_draft(
    question: Annotated[str, typer.Option(help="The prediction question, in plain words.")],
    connection: ConnectionOption = None,
) -> None:
    """Turn a question into a task (entity, target, horizon, cutoffs) to read, then confirm."""
    from fastapi import HTTPException

    from app.schemas.tasks import TaskDraftRequest, TaskSpecInput
    from app.services.llm_gateway import make_gateway
    from app.services.task_service import TaskService

    async def go() -> int:
        async with session() as db:
            pid = await project_id(db)
            conn = await pick_connection(db, pid, connection)
            service = TaskService(db)
            draft = await service.draft(
                pid,
                TaskDraftRequest(question=question, connection_id=conn.id),
                make_gateway(),
            )
            how = (
                "a language model"
                if draft.decision_mode == "llm"
                else "the offline rule-based drafter (no language model answered)"
            )
            if draft.status == "clarify":
                say(f"More detail is needed: {draft.clarifying_question}")
                return EXIT_NEEDS_ANSWER
            if draft.status == "invalid" or draft.yaml is None:
                for issue in draft.issues:
                    say(f"  - {issue.path}: {issue.message} ({issue.severity})")
                die("The draft still breaks the task format or your schema; ask it differently.")
            try:
                saved = await service.create(
                    pid,
                    TaskSpecInput(
                        yaml=draft.yaml, connection_id=conn.id, draft_source=draft.source
                    ),
                )
            except HTTPException as e:
                if e.status_code != 409:
                    raise
                die(
                    f"{detail_text(e.detail).split(';')[0]}. See `mlpilot task list`; confirm "
                    "that one with `mlpilot task confirm --task NAME`, or ask the question "
                    "in other words."
                )
        say(f"Drafted by {how}.")
        say()
        say(f"Task {saved.name!r} ({saved.id[:8]}), draft:")
        say(f"  {draft.description}")
        if draft.assumptions:
            say()
            say("Assumptions the draft made:")
            for a in draft.assumptions:
                say(f"  - {a}")
        warnings = [i for i in draft.issues if i.severity == "warning"]
        for w in warnings:
            say(f"warning: {w.path}: {w.message}")
        say()
        say("Next: `mlpilot task confirm` (shows the label counts first).")
        return 0

    code = run_async(go)
    if code:
        raise typer.Exit(code)


@task_app.command("list")
def task_list() -> None:
    """List the tasks of the project."""
    from sqlalchemy import select

    from app.db.models import TaskSpec

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            rows = (
                await db.scalars(
                    select(TaskSpec).where(TaskSpec.project_id == pid).order_by(TaskSpec.created_at)
                )
            ).all()
            table(
                [[r.id[:8], r.name, f"v{r.version}", r.status] for r in rows],
                ["id", "name", "version", "status"],
            )

    run_async(go)


@task_app.command("confirm")
def task_confirm(task: TaskOption = None) -> None:
    """Check the labels the task would train on, then confirm it (a run needs a confirmed task)."""
    from app.schemas.tasks import LabelPreviewRequest
    from app.services.task_service import TaskService

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            row = await pick_task(db, pid, task, "draft")
            service = TaskService(db)
            read = await service.get(pid, row.id)
            say(f"Task {read.name!r} ({read.id[:8]}), version {read.version}:")
            say(read.yaml.rstrip())
            say()
            say("Building the labels (read-only, point in time)...")
            preview = await service.preview_labels(pid, row.id, LabelPreviewRequest())
            table(
                [
                    [
                        day(c.cutoff),
                        f"{c.eligible:,}",
                        "-" if c.positives is None else f"{c.positives:,}",
                        number(c.base_rate),
                    ]
                    for c in preview.cutoffs
                ],
                ["cutoff", "eligible", "positives", "base rate"],
            )
            for dropped in preview.dropped_cutoffs:
                say(f"left out: {day(dropped.cutoff)} ({dropped.reason})")
            say(f"feasibility: {preview.feasibility.status}")
            for reason in preview.feasibility.reasons:
                say(f"  - {reason}")
            confirmed = await service.confirm(pid, row.id, None)
        say()
        say(f"Confirmed {confirmed.name!r} (version {confirmed.version}). Next: `mlpilot run`.")

    run_async(go)


# -- run -----------------------------------------------------------------------------------------


def _run_summary(state: Any) -> None:
    say(
        f"Run {state.id[:8]}: {state.status}"
        + (f" ({state.stop_reason})" if state.stop_reason else "")
    )
    say(f"  proposals: {state.rounds}, kept: {state.accepted}")
    if state.stop_reason == "no_llm":
        say(
            "  No language model answered, so no features were proposed: this is the automatic "
            "baseline only. Set an API key (see backend/.env.example) for the full loop."
        )
    if state.champion_val_metrics:
        val = state.champion_val_metrics
        say(
            f"  validation PR-AUC: {number(val.get('pr_auc'))} (base rate {number(val.get('base_rate'))})"
        )
    if state.test_metrics:
        test = state.test_metrics
        say(f"  test PR-AUC (scored once, at the end): {number(test.get('pr_auc'))}")
    if state.test_error:
        say(f"  test metric not available: {state.test_error}")
    if state.error:
        say(f"  error: {state.error}")


@app.command("run")
def run_command(
    task: TaskOption = None,
    max_rounds: Annotated[
        int, typer.Option(min=1, max=200, help="Features the model is asked for, at most.")
    ] = 20,
    patience: Annotated[
        int, typer.Option(min=1, help="Stop after this many rounds without a kept feature.")
    ] = 5,
    max_cost_usd: Annotated[
        float | None, typer.Option(min=0.0, help="Stop when the language model has cost this much.")
    ] = None,
    max_seconds: Annotated[
        float | None, typer.Option(min=0.0, help="Stop after this much wall time.")
    ] = None,
    snapshot: Annotated[
        str | None,
        typer.Option(help="Reuse this database snapshot (data version id) instead of a new one."),
    ] = None,
    override: Annotated[
        bool, typer.Option(help="Run even though the feasibility checks block the task.")
    ] = False,
) -> None:
    """Snapshot the data, build the baseline, then try features one at a time; keep what helps.

    The test rows are scored once, at the end. Without an API key the offline stub answers, and
    the run says so: it is then the automatic baseline only.
    """
    from app.db.models import Job
    from app.jobs import (
        handlers,  # noqa: F401  (registers the job kinds)
        runner,
    )
    from app.jobs.runner import JobWorker
    from app.schemas.runs import RunLoopRequest
    from app.schemas.snapshot import SnapshotRequest
    from app.schemas.tasks import RunStartRequest
    from app.services.llm_gateway import make_gateway
    from app.services.run_service import RunService
    from app.services.snapshot_service import SnapshotService, resolve_as_of
    from app.services.task_service import TaskService

    request = RunLoopRequest(
        max_rounds=max_rounds,
        patience=patience,
        max_cost_usd=max_cost_usd or None,
        max_seconds=max_seconds or None,
    )
    started = time.monotonic()

    def stamp(message: str) -> None:
        say(f"[{time.monotonic() - started:6.1f}s] {message}")

    async def go() -> int:
        async with session() as db:
            pid = await project_id(db)
            row = await pick_task(db, pid, task, "confirmed")
            if row.status != "confirmed":
                die(f"Task {row.name!r} is a draft: run `mlpilot task confirm` first")
            assert row.connection_id is not None
            providers = sorted(p for p in make_gateway().providers if p != "stub")
            stamp(f"task {row.name!r}; language model: {', '.join(providers) or 'offline stub'}")
            version = snapshot
            if version is None:
                stamp("taking a snapshot of the database (read-only)...")

                async def progress(message: str, fraction: float) -> None:
                    stamp(f"  snapshot {fraction:4.0%} {message}")

                taken = await SnapshotService(db).take(
                    pid,
                    row.connection_id,
                    SnapshotRequest(mode="snapshot", as_of=resolve_as_of(None)),
                    progress,
                )
                version = taken.id
                await db.commit()
                stamp(
                    f"snapshot {taken.short_hash}: {taken.n_rows:,} rows, as of {day(taken.as_of)}"
                )
            made = await TaskService(db).start_run(
                pid, row.id, RunStartRequest(data_version_id=version, override=override)
            )
            run_id = made.id
            started_job = await RunService(db).start(pid, run_id, request)
            await db.commit()
        runner.notify()
        stamp(f"run {run_id[:8]} queued")

        worker = JobWorker()
        await worker.start()
        try:
            await follow(pid, run_id, started_job.job_id, stamp)
        except (asyncio.CancelledError, KeyboardInterrupt):
            stamp("interrupted: asking the run to stop at its next step...")
            async with session() as db:
                job = await db.get(Job, started_job.job_id)
                if job is not None:
                    await runner.request_cancel(db, job)
            await wait_for_job(started_job.job_id, timeout=120)
            raise
        finally:
            await worker.stop(drain=10)
        async with session() as db:
            state = await RunService(db).state(pid, run_id)
        say()
        _run_summary(state)
        if state.status == "failed":
            return EXIT_FAILED
        say()
        say(f"Next: `mlpilot report --run {run_id[:8]}`")
        return 0

    code = run_async(go)
    if code:
        raise typer.Exit(code)


async def wait_for_job(job_id: str, timeout: float) -> str | None:
    from app.db.models import Job

    deadline = time.monotonic() + timeout
    while True:
        async with session() as db:
            job = await db.get(Job, job_id)
        if job is None or job.status in FINAL_JOB:
            return None if job is None else str(job.status)
        if time.monotonic() > deadline:
            return str(job.status)
        await asyncio.sleep(0.5)


async def follow(pid: str, run_id: str, job_id: str, stamp: Callable[[str], None]) -> None:
    """Print where the run stands until the run and its job have both ended."""
    from app.db.models import Job
    from app.services.run_service import RunService

    seen: tuple[Any, ...] = ()
    while True:
        async with session() as db:
            state = await RunService(db).state(pid, run_id)
            job = await db.get(Job, job_id)
        step = job.current_step if job else None
        now = (state.status, step, state.rounds, state.accepted)
        if now != seen:
            seen = now
            stamp(
                f"{state.status}: {step or 'waiting for a worker'}"
                + (f" (proposals {state.rounds}, kept {state.accepted})" if state.rounds else "")
            )
        # the job row is closed a moment after the run: wait for both so nothing is left "running"
        if state.status in FINAL_RUN and (job is None or job.status in FINAL_JOB):
            return
        await asyncio.sleep(1.0)


# -- report, export, score -----------------------------------------------------------------------


@app.command()
def report(
    run: RunOption = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="markdown, html (one file) or json.")
    ] = "markdown",
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write the report here (markdown: default is stdout)."),
    ] = None,
) -> None:
    """The evidence report of a run: question, data, split, every feature with its gain, the limits."""
    from app.services.report_service import ReportService

    if format not in ("markdown", "html", "json"):
        die("--format must be markdown, html or json")
    suffix = {"markdown": "md", "html": "html", "json": "json"}[format]

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            chosen = await pick_run(db, pid, run)
            text, _ = await ReportService(db).render(pid, chosen.id, format)  # type: ignore[arg-type]
        target = out or (None if format == "markdown" else Path(f"report-{chosen.id[:8]}.{suffix}"))
        if target is None:
            say(text)
            return
        target.write_text(text, encoding="utf-8")
        say(f"Report of run {chosen.id[:8]} written to {target} ({len(text):,} characters).")

    run_async(go)


@app.command()
def export(
    run: RunOption = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="The zip to write (default: ./<bundle>.zip).")
    ] = None,
    dialect: Annotated[
        str | None, typer.Option(help="SQL dialect of the features: duckdb or postgres.")
    ] = None,
) -> None:
    """Export a finished run: features as SQL and dbt, the model, and a read-only score.py."""
    from app.services.bundle_service import BundleService

    if dialect not in (None, "duckdb", "postgres"):
        die("--dialect must be duckdb or postgres")

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            chosen = await pick_run(db, pid, run)
            data, filename = await BundleService(db).bundle(
                pid,
                chosen.id,
                dialect,  # type: ignore[arg-type]
            )
        target = out or Path(filename)
        target.write_bytes(data)
        say(f"Export bundle of run {chosen.id[:8]} written to {target} ({len(data):,} bytes).")

    run_async(go)


@app.command()
def score(
    run: RunOption = None,
    cutoff: Annotated[
        datetime | None,
        typer.Option(help="Score the entities eligible at this time (default: the last full day)."),
    ] = None,
    top_k: Annotated[
        int, typer.Option(min=1, help="Size of the list the summary describes.")
    ] = 300,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the full ranked list here (CSV).")
    ] = None,
) -> None:
    """Rank the entities eligible now with a finished run's model, with reasons (reads the database)."""
    from app.services.score_service import ScoreService

    async def go() -> None:
        async with session() as db:
            pid = await project_id(db)
            chosen = await pick_run(db, pid, run)
            service = ScoreService(db)
            result = await service.score(pid, chosen.id, cutoff, top_k)
            source = service.csv(pid, chosen.id, result.csv.rsplit("/", 1)[-1].removesuffix(".csv"))
        target = out or Path(f"scores-{chosen.id[:8]}-{result.csv.rsplit('/', 1)[-1]}")
        shutil.copyfile(source, target)
        say(f"Scored {result.summary['n_scored']:,} entities at {day(result.cutoff)}.")
        for w in result.warnings:
            say(f"warning: {w['name']}: {w['detail']}")
        table(
            [
                [
                    str(r["rank"]),
                    str(r["entity_id"]),
                    number(r["score"]),
                    str(r.get("reason_1", "")),
                ]
                for r in result.preview[:10]
            ],
            ["rank", "entity", "score", "top reason"],
        )
        say(f"Full list: {target}")

    run_async(go)


# -- ui ------------------------------------------------------------------------------------------


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the page in your browser.")
    ] = True,
) -> None:
    """Start the API and the web UI (one process) on this machine."""
    import ipaddress

    import uvicorn

    from app.core.local_token import load_or_create_token
    from app.core.ui import ui_index

    if ui_index() is None:
        die(
            "This install has no built web UI (a source checkout runs it with `python start.py`; "
            "a pip or Docker install ships it)."
        )
    loopback = host == "localhost"
    with contextlib.suppress(ValueError):  # a host name that is not an address stays "not loopback"
        loopback = loopback or ipaddress.ip_address(host).is_loopback
    if not loopback:
        typer.secho(
            f"Warning: {host} is reachable from other machines. Anyone with the token can use "
            "your saved database connections. Publish the port on 127.0.0.1 only "
            "(docker run -p 127.0.0.1:8000:8000), or keep the default --host.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{port}/#token={load_or_create_token()}"
    say(f"MLPilot UI: {url}")
    say("The part after # is your local access token; it never reaches a server log.")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
