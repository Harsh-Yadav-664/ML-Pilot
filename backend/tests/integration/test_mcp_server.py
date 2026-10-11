"""The MCP server (#69): an MCP client talks to `mlpilot.mcp_server` over the SDK's in-memory
transport (the same protocol as stdio) on the DuckDB copy of the demo database, with the
offline stub provider. Skipped without the optional `mcp` package (CI installs it in the step
"MCP server")."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session

from app.services import llm_gateway
from mlpilot import cli, mcp_server
from tests.fixtures.api import create_project
from tests.fixtures.demo_db import demo_duckdb
from tests.fixtures.gateway import stub_gateway

QUESTION = "Which customers will stop ordering in the next 30 days?"
TOOLS = {"describe_schema", "draft_task", "confirm_task", "run_task", "get_report", "score"}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_gateway, "make_gateway", stub_gateway)


@pytest.fixture
async def project_id(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh project the server is pinned to (SQLite timestamps tie within a second)."""
    pid = await create_project(client)

    async def pinned(db: object) -> str:
        return pid

    monkeypatch.setattr(cli, "project_id", pinned)
    return pid


async def call(session: Any, tool: str, **arguments: Any) -> Any:
    result = await session.call_tool(tool, arguments)
    print(f"\n{tool}({json.dumps(arguments, default=str)}) -> error={result.isError}")
    text = result.content[0].text if result.content else ""
    print(text[:700])
    return result


async def test_a_client_lists_the_tools_and_runs_draft_task_and_run_task_on_the_demo_db(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    database = demo_duckdb(tmp_path / "demo.duckdb")
    made = await client.post(
        f"/api/v1/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "duckdb", "database": str(database)},
    )
    assert made.status_code in (200, 201), made.text

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as session:
        tools = (await session.list_tools()).tools
        print("tools:", sorted(t.name for t in tools))
        assert {t.name for t in tools} == TOOLS

        schema = (await call(session, "describe_schema")).structuredContent
        assert schema["connection"] == "shop"
        assert {"customers", "orders"} <= {t["table"] for t in schema["tables"]}

        drafted = await call(session, "draft_task", question=QUESTION)
        assert not drafted.isError
        assert drafted.structuredContent["status"] == "draft"
        assert drafted.structuredContent["decision_mode"] == "fallback"  # said, not hidden
        name = drafted.structuredContent["task"]

        # a draft cannot be run: confirming comes first
        refused = await call(session, "run_task", task=name)
        assert refused.isError and "confirmed" in refused.content[0].text

        confirmed = await call(session, "confirm_task", task=name, confirmed=True)
        assert not confirmed.isError and confirmed.structuredContent["status"] == "confirmed"
        assert confirmed.structuredContent["labels"]

        # the budget is enforced by the tool's schema, not by good manners
        over = await call(session, "run_task", task=name, max_features=1000)
        assert over.isError

        ran = await call(session, "run_task", task=name, max_features=1)
        assert not ran.isError, ran.content
        run = ran.structuredContent
        assert run["status"] == "completed" and run["budget"]["max_features"] == 1
        assert run["stop_reason"] == "no_llm" and run["proposals"] <= 1
        assert run["test_metrics"]["pr_auc"] is not None

        report = await call(session, "get_report", run=run["run_id"])
        assert "Leakage and safety" in report.content[0].text

        scored = await call(session, "score", run=run["run_id"])
        assert not scored.isError
        assert scored.structuredContent["summary"]["n_scored"] > 50
        assert not {"csv", "preview"} & set(scored.structuredContent)  # no database values


async def test_confirm_task_without_the_users_confirmation_is_refused(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    database = demo_duckdb(tmp_path / "demo.duckdb")
    await client.post(
        f"/api/v1/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "duckdb", "database": str(database)},
    )
    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as session:
        drafted = await call(session, "draft_task", question=QUESTION)
        name = drafted.structuredContent["task"]
        for arguments in ({"task": name}, {"task": name, "confirmed": False}):
            refused = await call(session, "confirm_task", **arguments)
            assert refused.isError and "explicit yes" in refused.content[0].text
        # still a draft: the refusals changed nothing
        tasks = (await client.get(f"/api/v1/projects/{project_id}/tasks/")).json()
        assert [t["status"] for t in tasks] == ["draft"]


async def test_the_mlpilot_mcp_command_speaks_mcp_on_stdio(tmp_path: Path) -> None:
    """The real process: nothing but protocol on stdout (the migration must not print there)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "MLPILOT_TOKEN")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mlpilot", "mcp"],
        env={**env, "MLPILOT_HOME": str(tmp_path)},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        assert {t.name for t in (await session.list_tools()).tools} == TOOLS


async def test_no_tool_takes_sql_or_connection_details() -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as session:
        for tool in (await session.list_tools()).tools:
            names = {tool.name, *tool.inputSchema.get("properties", {})}
            assert not any(
                word in n.lower() for n in names for word in ("sql", "query", "password", "dsn")
            ), (tool.name, names)
            print(tool.name, sorted(tool.inputSchema.get("properties", {})))
