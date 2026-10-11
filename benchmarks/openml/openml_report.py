"""``benchmarks/openml/RESULTS.md`` from the result files of real runs (issue #67).

The tables are computed from ``<results>/<id>-<name>/<method>_seed<n>.json`` and from nothing else:
no number is typed in. A method with no ok result on a dataset is shown as failed, with the reason.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from openml_harness import METHODS

Records = dict[str, list[dict[str, Any]]]  # method -> runs on one dataset


def collect(results: Path) -> list[dict[str, Any]]:
    records = [json.loads(p.read_text()) for p in sorted(results.glob("*/*_seed*.json"))]
    if not records:
        raise FileNotFoundError(f"no result files under {results}")
    return records


def mean_std(values: list[float], digits: int = 4) -> str:
    """``mean ± std`` (sample standard deviation); the std needs two or more seeds."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.{digits}f} (1 seed)"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def _ok(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in runs if r["status"] == "ok"]


def _by_seed(runs: list[dict[str, Any]], metric: str) -> dict[int, float]:
    return {int(r["seed"]): r["test"][metric] for r in _ok(runs)}


def _cell(runs: list[dict[str, Any]], metric: str, best: float | None) -> str:
    ok = _ok(runs)
    if not ok:
        why = sorted({str(r.get("error") or r["status"])[:80] for r in runs})
        return f"failed: {'; '.join(why)}" if runs else "not run"
    values = [r["test"][metric] for r in ok]
    text = mean_std(values)
    if best is not None and statistics.fmean(values) == best:
        text = f"**{text}**"
    return text + (f" ({len(runs) - len(ok)} failed)" if len(ok) < len(runs) else "")


def metric_table(by_dataset: dict[str, Records], metric: str) -> list[str]:
    methods = list(METHODS)
    lines = [
        "| Dataset | " + " | ".join(METHODS[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for name, runs in by_dataset.items():
        means = [
            statistics.fmean(_by_seed(runs[m], metric).values())
            for m in methods
            if _ok(runs.get(m, []))
        ]
        best = max(means) if means else None
        cells = [_cell(runs.get(m, []), metric, best) for m in methods]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def time_table(by_dataset: dict[str, Records]) -> list[str]:
    methods = list(METHODS)
    lines = [
        "| Dataset | " + " | ".join(METHODS[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for name, runs in by_dataset.items():
        cells = [
            f"{statistics.fmean(r['wall_seconds'] for r in _ok(runs[m])):.1f} s"
            if _ok(runs.get(m, []))
            else ""
            for m in methods
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def versus(by_dataset: dict[str, Records], other: str, metric: str) -> list[str]:
    """MLPilot against ``other``, seed by seed on the same split: wins per dataset and the gap."""
    lines: list[str] = []
    rows: list[tuple[str, float, float, int, int]] = []
    for name, runs in by_dataset.items():
        a, b = _by_seed(runs.get("mlpilot", []), metric), _by_seed(runs.get(other, []), metric)
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        diffs = [a[s] - b[s] for s in seeds]
        rows.append(
            (
                name,
                statistics.fmean(diffs),
                statistics.pstdev(diffs),
                sum(d > 0 for d in diffs),
                len(diffs),
            )
        )
    if not rows:
        return lines
    label = f"MLPilot minus {METHODS[other].split(' (')[0]}"
    lines += [
        f"| Dataset | {label}, mean ± std over seeds | Seeds where MLPilot is higher |",
        "|---|---|---|",
    ]
    for name, mean, std, wins, n in rows:
        lines.append(f"| {name} | {mean:+.4f} ± {std:.4f} | {wins} of {n} |")
    higher = sum(mean > 0 for _, mean, *_ in rows)
    lines.append(
        f"| **All {len(rows)} datasets** | mean {statistics.fmean(m for _, m, *_ in rows):+.4f} "
        f"| MLPilot higher on {higher} of {len(rows)} datasets (by the mean) |"
    )
    return lines


def llm_line(by_dataset: dict[str, Records]) -> str:
    runs = [r for d in by_dataset.values() for r in _ok(d.get("mlpilot", [])) if r.get("llm")]
    if not runs:
        return "- LLM: no mlpilot run."
    answered = [r for r in runs if r["llm"]["answered"]]
    accepted = sum(len(r["agent"]["accepted"]) for r in runs)
    if not answered:
        return (
            f"- LLM: none of the {len(runs)} mlpilot runs was answered by a model (offline stub): "
            f"no feature was proposed, the LLM cost is $0, and `mlpilot` is `lightgbm` run through "
            f"the loop. The comparison with AutoGluon below is the baseline's, not the agent's."
        )
    cost = sum(r["llm"]["cost_usd_estimated"] for r in runs)
    models = sorted({m for r in answered for m in r["llm"]["answered_by"]})
    return (
        f"- LLM: {len(answered)} of {len(runs)} mlpilot runs were answered by {', '.join(models)}; "
        f"{accepted} features accepted in total; estimated cost ${cost:.4f} in total "
        f"(the gateway counts words, not tokens)."
    )


def render(records: list[dict[str, Any]]) -> str:
    env = next(
        (r["environment"] for r in records if r["status"] == "ok"), records[0]["environment"]
    )
    groups: dict[str, Records] = {}
    meta: dict[str, dict[str, Any]] = {}
    for r in sorted(records, key=lambda r: str(r["dataset"]["name"])):
        d = r["dataset"]
        name = f"{d['name']} ({d['id']})"
        meta[name] = d
        groups.setdefault(name, defaultdict(list))[r["method"]].append(r)
    seeds = sorted({int(r["seed"]) for r in records})
    ag = next(
        (r["engine"] for r in records if r["method"] == "autogluon" and r["status"] == "ok"), None
    )
    lines = [
        "# OpenML single-table benchmark",
        "",
        (
            "Generated by `benchmarks/openml/openml_run.py --aggregate` from the result files of "
            "real runs; nothing in this file is typed in."
        ),
        "",
        (
            f"- Seeds {', '.join(map(str, seeds))}. Per dataset and seed ONE stratified 60/20/20 "
            "train / validation / test split, the same rows for every method (checked below); the "
            "test rows are scored once per run, after every fit."
        ),
        (
            "- PR-AUC is average precision, the positive class is the rarer one. Mean ± sample "
            "standard deviation over seeds; the best mean per row is bold. A difference smaller "
            "than the spread over seeds is not a win."
        ),
        "- LightGBM and MLPilot see the MLPilot preparation of the table and are refit on train + validation; "
        "AutoGluon is fitted on train + validation on the raw columns"
        + (
            f" (version {ag['version']}, preset `{ag['preset']}`, time limit {ag['time_limit_seconds']} s, "
            f"optimised for {ag['eval_metric']}; its own seed is its default)."
            if ag
            else "."
        )
        + (
            " PyTorch was not installed, so AutoGluon's neural-network models were not used."
            if ag and not env.get("torch")
            else ""
        ),
        llm_line(groups),
        f"- LightGBM {env['lightgbm']}, Python {env['python']}, {env.get('cpus')} CPUs; MLPilot commit "
        f"{env['mlpilot_commit'] or 'unknown'}"
        + (f", CI run {env['ci_run']}." if env.get("ci_run") else "."),
        "",
        "## Test PR-AUC",
        "",
        *metric_table(groups, "pr_auc"),
        "",
        "## Test ROC-AUC",
        "",
        *metric_table(groups, "roc_auc"),
        "",
        "## Wall time per run (fit, refit and scoring; the dataset download is not counted)",
        "",
        *time_table(groups),
        "",
    ]
    for metric, title in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
        for other in ("autogluon", "lightgbm"):
            table = versus(groups, other, metric)
            if table:
                lines += [
                    f"## {title}: MLPilot against {METHODS[other].split(' (')[0]}, same split per seed",
                    "",
                    *table,
                    "",
                ]
    mismatched = [
        name
        for name, runs in groups.items()
        for seed in seeds
        if len(
            {
                r["split_hash"]
                for m in runs.values()
                for r in m
                if r["seed"] == seed and r["status"] == "ok"
            }
        )
        > 1
    ]
    lines += [
        "## Datasets",
        "",
        "| Dataset | Rows | Features | Positive class (share) | Licence as read from OpenML |",
        "|---|---|---|---|---|",
        *(
            f"| [{n}]({m.get('url', '')}) | {m['n_rows']} | {m['n_features']} | {m['positive_class']} "
            f"({m['base_rate']:.1%}) | {m.get('licence') or 'not stated'} |"
            for n, m in meta.items()
        ),
        "",
        "Split check: "
        + (
            f"DIFFERENT SPLITS across methods for {', '.join(mismatched)}"
            if mismatched
            else "every method of a dataset and seed used the same train / validation / test rows."
        ),
        "",
    ]
    return "\n".join(lines)


def write(results: Path, out: Path) -> str:
    text = render(collect(results))
    out.write_text(text)
    return text
