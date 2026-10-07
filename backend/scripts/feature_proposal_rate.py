"""How many of a real model's feature proposals pass the checks on the demo churn task (#56).

    python -m scripts.feature_proposal_rate [count]

Builds the demo database's baseline (DFS features + LightGBM), then asks the model for ``count``
(default 20) features, one at a time, through the same proposer the product uses. Prints one line
per proposal and the pass rate, with the stage and reason of each rejection. The target of #56 is
70 percent of proposals passing the point-in-time guard and reaching execution.

It needs an API key; without one it says so and exits 0, because the offline stub gives no
proposals (the proposer then reports ``no_llm``). The CI job that runs it never blocks: the number
is a measurement, not a gate.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from collections import Counter
from pathlib import Path

from ai.context_builder import ContextBuilder
from ai.gateway import AIGateway
from app.core.config import settings
from ml.features.baseline import build_baseline
from ml.features.llm_sql import FeatureProposer
from ml.validation.splits import TemporalSplitPlan
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_dfs import world_for
from tests.unit.test_labels import CHURN

DEFAULT_COUNT = 20
TARGET = 0.70


async def main(count: int) -> int:
    gateway = AIGateway(settings)
    real = [p for p in gateway.list_providers() if p != "stub"]
    if not real:
        print("feature proposal rate: skipped (no LLM API key in the environment)")
        return 0
    print(f"providers with a key: {', '.join(real)}")
    with tempfile.TemporaryDirectory() as tmp:
        world = world_for(demo_duckdb(Path(tmp) / "demo.duckdb"), CHURN)
    plan = TemporalSplitPlan.from_spec(world["spec"])
    baseline = build_baseline(world["spec"], world["graph"], world["tables"], world["labels"], plan)
    proposer = FeatureProposer(
        spec=world["spec"],
        graph=world["graph"],
        tables=world["tables"],
        baseline=baseline,
        gateway=gateway,
        builder=ContextBuilder(),
    )
    records = await proposer.run(count)
    stages: Counter[str] = Counter()
    for r in records:
        name = r.proposal.name if r.proposal else "(no usable answer)"
        mark = "PASS" if r.status == "proposed" else "FAIL"
        suffix = " repaired" if r.repaired else ""
        print(f"{mark}  {name} [{r.status}{suffix}]")
        if r.status != "proposed":
            stages[r.stage or r.status] += 1
            print(f"      {r.stage}: {r.reasons[0] if r.reasons else ''}")
    passed = sum(r.status == "proposed" for r in records)
    first_try = sum(r.status == "proposed" and not r.repaired for r in records)
    if not records or records[0].status == "no_llm":
        print("no proposals: the model fell back (see the gateway log)")
        return 0
    print(
        f"{passed} of {len(records)} proposals passed all checks ({passed / len(records):.0%}; "
        f"target {TARGET:.0%}); {first_try} on the first try"
    )
    if stages:
        print("rejections by stage: " + ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT)))
