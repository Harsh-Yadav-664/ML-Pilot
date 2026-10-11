"""One benchmark run, or the table of all of them (issue #65).

    python benchmarks/relbench/run.py --dataset rel-f1 --task driver-dnf --method b --seed 0
    python benchmarks/relbench/run.py --aggregate

The first form writes ``benchmarks/results/<dataset>/<task>/<method>_seed<seed>.json``. A run that
fails writes a ``failed`` record with the error and exits non-zero. ``make benchmark`` loops over
methods and seeds, then aggregates into ``benchmarks/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "backend")]

import relbench_harness as h
import relbench_report as report

RESULTS = REPO / "benchmarks" / "results"
RESULTS_MD = REPO / "benchmarks" / "RESULTS.md"


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--aggregate", action="store_true", help="write RESULTS.md from the results")
    p.add_argument("--dataset", default="rel-f1", help="a RelBench dataset name or a local path")
    p.add_argument("--task", default="driver-dnf")
    p.add_argument("--method", default="b", choices=[*h.METHODS, *h.ALIASES])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", help="provider:model for methods c and d (else from the environment)")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=1800.0)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--markdown", type=Path, default=RESULTS_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.aggregate:
        print(report.write(args.results_dir, args.markdown))
        return 0
    method = h.ALIASES.get(args.method, args.method)
    model = args.model or os.environ.get(h.ENV_STRONG_MODEL if method == "d" else h.ENV_CHEAP_MODEL)
    settings = h.Settings(
        seed=args.seed,
        max_rounds=args.max_rounds,
        patience=args.patience,
        max_cost_usd=args.max_cost_usd,
        max_seconds=args.max_seconds,
        model=model or None,
    )
    dataset, task = h.load_task(args.dataset, args.task)
    bench = h.translate(dataset, task)
    if method == "d" and not h.llm_ready(model):
        reason = (
            f"no strong model: set {h.ENV_STRONG_MODEL} (provider:model) and that provider's key"
        )
        record = h.not_run(bench, method, settings, "skipped", reason=reason)
    else:
        try:
            record = asyncio.run(h.run_method(bench, method, settings))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:2000]
            failed = h.not_run(bench, method, settings, "failed", error=error)
            print(f"failed: {h.write_result(args.results_dir, failed)}", file=sys.stderr)
            raise
    path = h.write_result(args.results_dir, record)
    print(f"{record['status']}: {path}")
    if record["status"] == "ok":
        print(f"test {record['test']}  val {record['val']}  {record['wall_seconds']['total']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
