"""The published benchmark table (#66) is consistent with itself and with the README.

The numbers come from a CI run (benchmarks/RESULTS.md links it); this cannot re-run the benchmark.
It catches a mistyped or half-edited number: each row's mean and std must follow from the per-seed
scores printed beside them, and the README may only show 'ours' numbers that RESULTS.md has.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = (REPO / "benchmarks" / "RESULTS.md").read_text()
README = (REPO / "README.md").read_text()
RUN_URL = "https://github.com/Harsh-Yadav-664/ML-Pilot/actions/runs/"
ROW = re.compile(
    r"^\| ([abcd]): [^|]*\| (\d\.\d{4}) ± (\d\.\d{4}) \|[^|]*\| \d+ \(([^)]*)\) \|", re.MULTILINE
)


def _benchmark_section() -> str:
    return README.split("## Benchmark")[1].split("\n## ")[0]


def test_each_mean_and_std_follows_from_the_per_seed_scores() -> None:
    rows = ROW.findall(RESULTS)
    assert {m for m, *_ in rows} >= {"a", "b"}
    for method, mean, std, seeds in rows:
        scores = [float(s) for s in re.findall(r"\d: (\d\.\d{4})", seeds)]
        assert len(scores) >= 2, method
        # the per-seed scores are rounded to 4 decimals, so allow for that
        assert abs(statistics.fmean(scores) - float(mean)) < 2e-4, method
        assert abs(statistics.stdev(scores) - float(std)) < 2e-4, method


def test_the_readme_shows_only_ours_numbers_that_results_has() -> None:
    ours = re.findall(r"\| (\d\.\d{4} ± \d\.\d{4}) \| ours", _benchmark_section())
    assert len(ours) == 2
    for value in ours:
        assert value in RESULTS


def test_ours_links_to_a_ci_run_and_every_quoted_number_to_a_paper() -> None:
    table = _benchmark_section()
    run = re.search(re.escape(RUN_URL) + r"\d+", RESULTS)
    assert run and run.group(0) in table
    quoted = [line for line in table.splitlines() if "| quoted" in line]
    assert len(quoted) == 4
    for line in quoted:
        assert "arxiv.org/abs/" in line, line
