"""Prompt text for the feature proposer (#56). The text lives in files so it can be read and diffed."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def system_prompt() -> str:
    return (_HERE / "proposer_system.txt").read_text().strip()
