"""One benchmark run, or the table of all of them (issue #67).

    python benchmarks/openml/openml_run.py --dataset 37 --method lightgbm --seed 0
    python benchmarks/openml/openml_run.py --csv data.csv --target churn --method mlpilot
    python benchmarks/openml/openml_run.py --aggregate

The first forms write ``benchmarks/openml/results/<id>-<name>/<method>_seed<seed>.json``. A run that
fails writes a ``failed`` record with the error and exits non-zero. ``make benchmark-openml`` loops
over datasets, methods and seeds, then aggregates into ``benchmarks/openml/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# the RelBench harness is imported for its gateway helpers; both are script directories
sys.path[:0] = [str(HERE), str(REPO / "benchmarks" / "relbench"), str(REPO / "backend")]

import openml_harness as h
import openml_report as report

RESULTS = HERE / "results"
RESULTS_MD = HERE / "RESULTS.md"


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--aggregate", action="store_true", help="write RESULTS.md from the results")
    p.add_argument("--dataset", type=int, choices=sorted(h.DATASETS), help="an OpenML dataset id")
    p.add_argument("--csv", type=Path, help="a local CSV instead of an OpenML dataset")
    p.add_argument("--target", help="the target column of --csv")
    p.add_argument("--method", default="lightgbm", choices=list(h.METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model", help=f"provider:model for mlpilot (else ${h.ENV_MODEL}, else the stub)"
    )
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--ag-seconds", type=int, default=90, help="AutoGluon time limit per run")
    p.add_argument("--ag-preset", default="high_quality")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--markdown", type=Path, default=RESULTS_MD)
    args = p.parse_args(argv)
    if not args.aggregate and (args.dataset is None) == (args.csv is None):
        p.error("give exactly one of --dataset and --csv")
    if args.csv and not args.target:
        p.error("--csv needs --target")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.aggregate:
        print(report.write(args.results_dir, args.markdown))
        return 0
    model = args.model or os.environ.get(h.ENV_MODEL) or None
    settings = h.Settings(
        seed=args.seed,
        max_rounds=args.max_rounds,
        patience=args.patience,
        max_cost_usd=args.max_cost_usd,
        max_seconds=args.max_seconds,
        model=model if args.method == "mlpilot" else None,
        ag_seconds=args.ag_seconds,
        ag_preset=args.ag_preset,
    )
    data = h.load_csv(args.csv, args.target) if args.csv else h.load_openml(args.dataset)
    try:
        record = asyncio.run(h.run_method(data, args.method, settings))
    except Exception as e:
        error = f"{type(e).__name__}: {e}"[:2000]
        failed = h.not_run(data, args.method, settings, "failed", error=error)
        print(f"failed: {h.write_result(args.results_dir, failed)}", file=sys.stderr)
        raise
    path = h.write_result(args.results_dir, record)
    t = record["test"]
    print(
        f"{record['status']}: {path.name}  {data.meta['name']} {args.method} seed {args.seed}  "
        f"test PR-AUC {t['pr_auc']:.4f} ROC-AUC {t['roc_auc']:.4f}  {record['wall_seconds']} s"
    )
    if record["llm"]:
        agent = record["agent"]
        print(
            f"llm {record['llm']}; agent stopped: {agent['stop_reason']}, accepted {agent['accepted']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
