"""The `mlpilot` command line (#68): the same services as the API, driven the way a user does.

One test walks connect, schema, task draft, task confirm, run, report, export and score on a
DuckDB copy of the demo database with the offline stub provider; the others cover the refusals
and that every command in the README's block parses with the real command line. The commands run
in a worker thread because each one starts its own event loop (as the real command does).
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import shlex
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import typer.main
from typer.testing import CliRunner, Result

from app import main as app_main
from app.services import llm_gateway
from mlpilot import cli
from tests.fixtures.demo_db import demo_duckdb
from tests.fixtures.gateway import stub_gateway

REPO = Path(__file__).resolve().parents[3]
QUESTION = "Which customers will stop ordering in the next 30 days?"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metadata database is already migrated; the LLM is the offline stub, whatever .env says."""
    monkeypatch.setattr(cli, "migrate", lambda: None)
    monkeypatch.setattr(llm_gateway, "make_gateway", stub_gateway)


async def mlpilot(*args: str, expect: int = 0) -> Result:
    result = await asyncio.to_thread(CliRunner().invoke, cli.app, list(args))
    print(f"\n$ mlpilot {shlex.join(args)}\n{result.output}")
    assert result.exit_code == expect, (result.exit_code, result.output, result.exception)
    return result


def readme_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("readme_cli", REPO / "scripts" / "readme_cli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_the_documented_commands_take_a_database_to_a_report_export_and_scores(
    project_id: str, tmp_path: Path
) -> None:
    database = demo_duckdb(tmp_path / "demo.duckdb")

    out = (
        await mlpilot(
            "connect", "--name", "shop", "--dialect", "duckdb", "--database", str(database)
        )
    ).output
    assert "Connected to 'shop'" in out and "The role is read-only." in out

    out = (await mlpilot("schema")).output
    assert "shop (duckdb): 9 tables" in out and "customers" in out and "declared" in out

    out = (await mlpilot("task", "draft", "--question", QUESTION)).output
    assert "offline rule-based drafter" in out  # a fallback is said out loud, never hidden
    assert "Task 'no_orders_30d'" in out and "labelled 1" in out

    out = (await mlpilot("task", "confirm")).output
    assert "cutoff" in out and "base rate" in out and "feasibility: ok" in out
    assert "Confirmed 'no_orders_30d'" in out

    out = (await mlpilot("run", "--max-rounds", "1")).output
    assert "completed (no_llm)" in out and "automatic baseline only" in out
    assert "test PR-AUC (scored once, at the end):" in out

    report = tmp_path / "report.md"
    await mlpilot("report", "--out", str(report))
    text = report.read_text()
    assert "run report" in text and "Leakage and safety" in text and "| PR-AUC |" in text
    html = tmp_path / "report.html"
    await mlpilot("report", "--format", "html", "--out", str(html))
    assert "<h1>" in html.read_text() and "<script" not in html.read_text()

    bundle = tmp_path / "bundle.zip"
    await mlpilot("export", "--out", str(bundle))
    names = zipfile.ZipFile(bundle).namelist()
    assert any(n.endswith("/score.py") for n in names) and any(
        n.endswith("model.txt") for n in names
    )

    ranked = tmp_path / "scores.csv"
    out = (await mlpilot("score", "--out", str(ranked))).output
    assert "Scored" in out and "Full list:" in out
    rows = list(csv.reader(ranked.open()))
    assert rows[0] == ["entity_id", "score", "rank", "decile", "reason_1", "reason_2", "reason_3"]
    scores = [float(r[1]) for r in rows[1:]]
    assert len(scores) > 50 and scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


async def test_a_run_started_by_the_command_is_what_the_api_and_the_ui_read(
    client: httpx.AsyncClient, project_id: str, tmp_path: Path
) -> None:
    """The command line and the web UI share one state: no second copy of anything."""
    database = demo_duckdb(tmp_path / "demo.duckdb")
    await mlpilot("connect", "--name", "shared", "--dialect", "duckdb", "--database", str(database))
    seen = (await client.get(f"/api/v1/projects/{project_id}/connections/")).json()
    assert [c["name"] for c in seen] == ["shared"]
    assert seen[0]["can_write"] is False


async def test_a_refusal_is_a_message_and_a_failing_exit_code_not_a_traceback(
    project_id: str, tmp_path: Path
) -> None:
    out = (await mlpilot("schema", expect=1)).output
    assert "Error: There is no connection yet. Add one with `mlpilot connect`." in out
    out = (await mlpilot("run", expect=1)).output
    assert "Error: There is no confirmed task yet" in out
    out = (await mlpilot("report", expect=1)).output
    assert "Error: There is no run yet" in out
    out = (
        await mlpilot(
            "connect",
            "--name",
            "gone",
            "--dialect",
            "sqlite",
            "--database",
            str(tmp_path / "missing.sqlite"),
            expect=1,
        )
    ).output
    assert "is not a readable sqlite database file" in out


async def test_a_password_is_taken_from_the_environment_never_the_command_line() -> None:
    out = (await mlpilot("connect", "--help")).output
    assert "--password-env" in out and "--password " not in out
    out = (
        await mlpilot(
            "connect",
            "--name",
            "pg",
            "--host",
            "localhost",
            "--database",
            "d",
            "--user",
            "u",
            "--password-env",
            "MLPILOT_TEST_NO_SUCH_VARIABLE",
            expect=1,
        )
    ).output
    assert "MLPILOT_TEST_NO_SUCH_VARIABLE is not set" in out


async def test_every_command_in_the_readme_block_parses_with_the_real_command_line() -> None:
    script = readme_script()
    lines = script.commands()
    assert script.main(["--check"]) == 0, lines
    root = typer.main.get_command(cli.app)
    parsed = []
    for line in (ln for ln in lines if ln.startswith("mlpilot ")):
        rest = shlex.split(line)[1:]
        command: Any = root
        while hasattr(command, "commands"):  # a group: the next word names the subcommand
            command = command.commands[rest.pop(0)]
        # parsing the options is what fails on a typo or a missing required option
        ctx = command.make_context(command.name or "", rest)
        parsed.append((command.name, sorted(ctx.params)))
    print(parsed)
    assert [name for name, _ in parsed] == ["connect", "draft", "confirm", "run", "report"]


async def test_the_command_line_shows_its_version() -> None:
    out = (await mlpilot("--version")).output
    assert out.strip() == f"mlpilot {cli.__version__}"


async def test_the_ui_command_refuses_to_start_without_a_built_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.core.ui.ui_index", lambda: None)
    out = (await mlpilot("ui", "--no-open", expect=1)).output
    assert "no built web UI" in out


async def test_the_root_serves_the_built_ui_when_there_is_one_and_the_name_when_not(
    client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_main, "ui_index", lambda: None)
    plain = await client.get("/")
    assert plain.json()["app"] == "MLPilot"

    page = tmp_path / "index.html"
    page.write_text('<!doctype html><div id="root"></div>')
    monkeypatch.setattr(app_main, "ui_index", lambda: page)
    served = await client.get("/")
    assert served.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in served.text and served.headers["cache-control"] == "no-store"
    # serving the page opens nothing else: the API still wants the token
    anonymous = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_main.app), base_url="http://test"
    )
    async with anonymous:
        assert (await anonymous.get("/api/v1/projects/")).status_code == 401
