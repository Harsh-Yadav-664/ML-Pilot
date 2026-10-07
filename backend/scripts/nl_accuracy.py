"""How often the language model drafts the expected spec for the demo database's questions (#53).

    python -m scripts.nl_accuracy

Runs the 10 questions of tests/fixtures/nl_questions.yaml through ``draft_spec`` with whatever
providers the environment has keys for, using the default (cheap) tier, and prints one line per
question and the count. It needs an API key; without one it says so and exits 0, because the
offline stub cannot answer these (its fallback is the rule-based drafter, tested separately).
The CI job that runs it is not required to pass: the number is a measurement, not a gate.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ai.context_builder import ContextBuilder
from ai.gateway import AIGateway
from app.core.config import settings
from ml.data.schema_graph import build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_nl_to_spec import FIXTURE, matches

from ml.tasks import nl_to_spec  # isort: skip

AS_OF = datetime(2025, 1, 1, tzinfo=UTC)


async def main() -> int:
    gateway = AIGateway(settings)
    real = [p for p in gateway.list_providers() if p != "stub"]
    if not real:
        print("nl accuracy: skipped (no LLM API key in the environment)")
        return 0
    print(f"providers with a key: {', '.join(real)}")
    cases = yaml.safe_load(FIXTURE.read_text())["questions"]
    with tempfile.TemporaryDirectory() as tmp:
        path = demo_duckdb(Path(tmp) / "demo.duckdb")
        graph = build_schema_graph(
            open_source(ConnectionSpec(dialect="duckdb", database=str(path)))
        )
    hits = 0
    for case in cases:
        draft = await nl_to_spec.draft_spec(
            case["q"], graph, gateway=gateway, builder=ContextBuilder(), as_of=AS_OF
        )
        if draft.status != "spec" or draft.spec is None:
            problem = draft.clarifying_question or "; ".join(i.message for i in draft.errors)
            print(f"MISS  {case['q']}\n      {draft.status} ({draft.decision_mode}): {problem}")
            continue
        diff = matches(draft.spec, case["expect"])
        hits += not diff
        mark = "OK  " if not diff else "MISS"
        suffix = f" repaired={draft.repaired}" if draft.repaired else ""
        print(f"{mark}  {case['q']} [{draft.decision_mode}{suffix}]")
        for line in diff:
            print(f"      {line}")
    print(f"{hits} of {len(cases)} questions drafted the expected spec")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
