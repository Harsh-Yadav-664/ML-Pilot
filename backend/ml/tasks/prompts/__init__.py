"""Prompt text for the task drafter (#53). The text lives in files so it can be read and diffed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).parent


def system_prompt() -> str:
    return (_HERE / "spec_system.txt").read_text().strip()


def worked_examples() -> list[dict[str, Any]]:
    """Three worked examples: a question, the spec answer (YAML) and the assumptions."""
    data = yaml.safe_load((_HERE / "spec_examples.yaml").read_text())
    return list(data["examples"])
