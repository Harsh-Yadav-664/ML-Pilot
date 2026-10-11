"""MLPilot as an MCP server, so an AI assistant can use it as a tool (issue #69).

    pip install -r requirements-mcp.txt   # the optional ``mcp`` SDK (MIT)
    mlpilot mcp                           # stdio; point the assistant's MCP config at this command

The tools call the same services as the ``mlpilot`` commands and the HTTP API, in this process
(``mlpilot.cli`` has the shared helpers), so every guard applies unchanged: SQL goes through the
SQL guard and the point-in-time guard, the test rows are scored once, at the end, and the
language model is the one the gateway is configured with (the offline stub without keys).

Deliberately not here: a tool that runs SQL, a tool that saves a connection (a password would pass
through the assistant; use ``mlpilot connect``), and anything that writes to the user's database.
Rows that ``score`` returns go to the assistant: ask for a short list.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from mlpilot import cli

DEFAULT_MAX_FEATURES = 5  # a run is a budgeted call: small by default, capped by the schema below
DEFAULT_MAX_SECONDS = 600.0

mcp = FastMCP(
    "mlpilot",
    instructions=(
        "MLPilot answers prediction questions on a relational database. Order: describe_schema, "
        "draft_task, show the draft to the user, confirm_task only after they agree, run_task, "
        "then get_report and score. It reads the database read-only and never runs SQL you write."
    ),
)


@contextmanager
def refusals() -> Iterator[None]:
    """A refused request is a tool error with the service's reasons, not a traceback."""
    try:
        yield
    except HTTPException as e:
        raise ToolError(cli.detail_text(e.detail)) from None
    except cli.NotFound as e:
        raise ToolError(str(e)) from None


@mcp.tool()
async def describe_schema(connection: str | None = None) -> dict[str, Any]:
    """Tables, time columns and relationships of a saved connection (names and types only, no
    rows). `connection` is a name or id; the default is the latest."""
    from app.services.connection_service import ConnectionService

    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            conn = await cli.pick_connection(db, pid, connection)
            graph = await ConnectionService(db).schema_graph(conn)
    return {
        "connection": conn.name,
        "dialect": graph.dialect,
        "tables": [
            {
                "table": t.key,
                "rows": t.row_count,
                "time_column": t.time_column,
                "primary_key": t.primary_key,
                "columns": [{"name": c.name, "type": c.type, "hint": c.hint} for c in t.columns],
            }
            for t in graph.tables
        ],
        "relationships": [
            f"{e.from_table}({', '.join(e.from_columns)}) -> {e.to_table}({', '.join(e.to_columns)})"
            for e in graph.edges
        ],
        "warnings": graph.warnings,
    }


@mcp.tool()
async def draft_task(question: str, connection: str | None = None) -> dict[str, Any]:
    """Turn a prediction question into a task (entity, target, horizon, cutoffs) and save it as a
    draft. Show the draft to the user: it is not used until confirm_task."""
    from app.schemas.tasks import TaskDraftRequest, TaskSpecInput
    from app.services.llm_gateway import make_gateway
    from app.services.task_service import TaskService

    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            conn = await cli.pick_connection(db, pid, connection)
            service = TaskService(db)
            draft = await service.draft(
                pid, TaskDraftRequest(question=question, connection_id=conn.id), make_gateway()
            )
            if draft.status == "clarify":
                return {"status": "clarify", "ask_the_user": draft.clarifying_question}
            if draft.status == "invalid" or draft.yaml is None:
                found = "; ".join(f"{i.path}: {i.message}" for i in draft.issues)
                raise ToolError(f"The draft breaks the task format or the schema: {found}")
            saved = await service.create(
                pid,
                TaskSpecInput(yaml=draft.yaml, connection_id=conn.id, draft_source=draft.source),
            )
    return {
        "status": "draft",
        "task": saved.name,
        "task_id": saved.id,
        "description": draft.description,
        "assumptions": draft.assumptions,
        "warnings": [f"{i.path}: {i.message}" for i in draft.issues if i.severity == "warning"],
        "decision_mode": draft.decision_mode,  # "fallback": the offline rule-based drafter
        "yaml": draft.yaml,
    }


@mcp.tool()
async def confirm_task(task: str | None = None, confirmed: bool = False) -> dict[str, Any]:
    """Confirm a drafted task so it can be run. Refused unless `confirmed` is true, which you
    may only set after the user has read the draft and said yes. Returns the label counts."""
    from app.schemas.tasks import LabelPreviewRequest
    from app.services.task_service import TaskService

    if confirmed is not True:
        raise ToolError(
            "Refused: confirming a task needs the user's explicit yes. Show them the draft, and "
            "call again with confirmed=true once they agree."
        )
    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            row = await cli.pick_task(db, pid, task, "draft")
            service = TaskService(db)
            preview = await service.preview_labels(pid, row.id, LabelPreviewRequest())
            done = await service.confirm(pid, row.id, None)
    return {
        "task": done.name,
        "status": done.status,
        "labels": [
            {
                "cutoff": cli.day(c.cutoff),
                "eligible": c.eligible,
                "positives": c.positives,
                "base_rate": c.base_rate,
            }
            for c in preview.cutoffs
        ],
        "feasibility": preview.feasibility.status,
        "feasibility_reasons": preview.feasibility.reasons,
    }


@mcp.tool()
async def run_task(
    task: str | None = None,
    max_features: Annotated[int, Field(ge=1, le=20)] = DEFAULT_MAX_FEATURES,
    max_seconds: Annotated[float, Field(ge=10, le=3600)] = DEFAULT_MAX_SECONDS,
) -> dict[str, Any]:
    """Run a confirmed task and wait for it: snapshot the data, build the baseline, try at most
    `max_features` features one at a time and keep what helps on validation, within `max_seconds`.
    The test rows are scored once, at the end. Without an API key the offline stub answers and
    only the automatic baseline is built (stop_reason "no_llm")."""
    from app.schemas.runs import RunLoopRequest
    from app.services.run_service import RunService

    request = RunLoopRequest(max_rounds=max_features, max_seconds=max_seconds)
    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            row = await cli.pick_task(db, pid, task, "confirmed")
        run_id = await cli.execute_run(pid, row, request, None, False, lambda message: None)
        async with cli.session() as db:
            state = await RunService(db).state(pid, run_id)
    if state.status == "failed":
        raise ToolError(f"Run {run_id} failed: {state.error}")
    return {
        "run_id": run_id,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "budget": {"max_features": max_features, "max_seconds": max_seconds},
        "proposals": state.rounds,
        "kept": state.accepted,
        "validation_metrics": state.champion_val_metrics,
        "test_metrics": state.test_metrics,
        "test_error": state.test_error,
    }


@mcp.tool()
async def get_report(
    run: str | None = None, format: Literal["markdown", "json"] = "markdown"
) -> str:
    """The evidence report of a run: question, data, split, every feature with its gain, and the
    limits. `run` is an id (or its first characters); the default is the latest run."""
    from app.services.report_service import ReportService

    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            chosen = await cli.pick_run(db, pid, run)
            text, _ = await ReportService(db).render(pid, chosen.id, format)
    return text


@mcp.tool()
async def score(run: str | None = None, cutoff: datetime | None = None) -> dict[str, Any]:
    """Score the entities eligible now with a finished run's model. Returns counts, drift
    warnings and the score range only, never rows: entity ids and reasons are values from the
    database, which an assistant does not see. `mlpilot score --out` writes the ranked list.
    `cutoff` defaults to the last full day."""
    from app.services.score_service import ScoreService

    with refusals():
        async with cli.session() as db:
            pid = await cli.project_id(db)
            chosen = await cli.pick_run(db, pid, run)
            result = await ScoreService(db).score(pid, chosen.id, cutoff, 1)
    return result.model_dump(mode="json", exclude={"csv", "preview"})


def serve() -> None:
    """Serve the tools on stdin and stdout until the assistant disconnects."""
    mcp.run(transport="stdio")
