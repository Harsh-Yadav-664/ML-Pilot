"""The user docs (#70) point at things that exist: files, headings and `mlpilot` commands."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlpilot.cli import app

DOCS = Path(__file__).resolve().parents[3] / "docs"
PAGES = ["quickstart.md", "safety.md"]
LINK = re.compile(r"\]\((?!https?:|mailto:)([^)#\s]*)(?:#([^)\s]*))?\)")


def _anchor(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.lower()).strip().replace(" ", "-")


@pytest.mark.parametrize("page", PAGES)
def test_every_relative_link_and_heading_anchor_exists(page: str) -> None:
    for path, anchor in LINK.findall((DOCS / page).read_text()):
        target = (DOCS / page).parent / path if path else DOCS / page
        assert target.resolve().exists(), f"{page}: {path} does not exist"
        if anchor:
            headings = re.findall(r"^#+ (.*)$", target.read_text(), re.MULTILINE)
            assert anchor in {_anchor(h) for h in headings}, f"{page}: no heading #{anchor}"


def test_every_mlpilot_command_in_the_quickstart_exists() -> None:
    commands = re.findall(
        r"^(?:\s*)mlpilot ((?:task )?[a-z]+)", (DOCS / "quickstart.md").read_text(), re.MULTILINE
    )
    assert {
        "connect",
        "schema",
        "task draft",
        "task confirm",
        "run",
        "report",
        "export",
        "score",
    } <= set(commands)
    for command in commands:
        result = CliRunner().invoke(app, [*command.split(), "--help"])
        assert result.exit_code == 0, f"mlpilot {command}: {result.output}"
