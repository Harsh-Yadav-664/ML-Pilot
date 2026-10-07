"""The evidence report of a run (#60): every number comes from a stored record, and the HTML
opens offline."""

# ruff: noqa: F811
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select

import app.db.session as db_session
from app.db.models import DataVersion, Experiment, Feature, Run, TaskSpec
from app.jobs import handlers
from ml.experiments.acceptance import DEFAULT_RULE
from ml.reports import build_report, to_html, to_markdown
from ml.reports.numbers import FORMATS, find_numbers, lookup
from tests.fixtures.api import API
from tests.integration.test_baseline_run_api import TABLES, connection, started_run  # noqa: F401
from tests.integration.test_run_loop_api import GOOD, finished, runs_url
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer


async def a_finished_run(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    accept_all: bool = False,
) -> str:
    """Three good proposals and one the guard refuses. With the default rule none of the three
    helps on the demo data (not kept); with ``accept_all`` the rule is loosened so all are kept."""
    if accept_all:
        monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)
        monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in GOOD] + [answer("leaky_sql")], cost=0.01)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    started = await client.post(
        runs_url(project_id, f"/{run_id}/start"),
        json={"max_rounds": 4, "patience": 4, "max_cost_usd": 0.5},
    )
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job
    return run_id


def expected(key: str, db: dict[str, Any]) -> float:
    """What the database holds for a number the report showed, read without the report's code."""
    run, final, base, proposals = db["run"], db["final"], db["baseline"], db["proposals"]
    calls = [c for f in proposals for c in (f.guard_results or {}).get("llm", [])]
    m = re.fullmatch(r"final\.test_metrics\.(\w+)", key)
    if m:
        return float(final.test_metrics[m[1]])
    m = re.fullmatch(r"final\.val_metrics\.(\w+)", key)
    if m:
        return float(final.val_metrics[m[1]])
    m = re.fullmatch(r"baseline\.experiment\.val_metrics\.(\w+)", key)
    if m:
        return float(base.val_metrics[m[1]])
    m = re.fullmatch(r"features\[(\d+)\]\.gain\.(\w+)(?:\[(\d+)\])?", key)
    if m:
        value = proposals[int(m[1])].gain[m[2]]
        return float(value[int(m[3])] if m[3] is not None else value)
    m = re.fullmatch(r"llm\.calls\[(\d+)\]\.(\w+)", key)
    if m:
        return float(calls[int(m[1])][m[2]])
    m = re.fullmatch(r"baseline\.features\[(\d+)\]\.importance", key)
    if m:
        ranked = sorted(db["dfs"], key=lambda f: -float(f.gain["importance"]))
        return float(ranked[int(m[1])].gain["importance"])
    m = re.fullmatch(r"champions\[(\d+)\]\.(round|val_pr_auc)", key)
    if m:
        champs = sorted(db["champions"], key=lambda e: e.manifest["round"])
        e = champs[int(m[1])]
        return float(e.manifest["round"] if m[2] == "round" else e.val_metrics["pr_auc"])
    m = re.fullmatch(r"labels\.cutoffs\[(\d+)\]\.(eligible|positives|base_rate)", key)
    if m:
        return float(db["cutoffs"][int(m[1])][m[2]])
    m = re.fullmatch(r"data\.tables\[(\d+)\]\.(rows|null_time_rows)", key)
    if m:
        tables = sorted(db["version"].source["tables"].items())
        return float(tables[int(m[1])][1][m[2]])
    statuses = [f.status for f in proposals]
    simple: dict[str, float] = {
        "run.seed": run.seed,
        "run.budget_used.cost_usd": run.budget_used["cost_usd"],
        "run.budget.max_cost_usd": run.budget["max_cost_usd"],
        "run.split_plan.folds": run.split_plan["folds"],
        "task.version": db["spec"].version,
        "data.n_rows": db["version"].n_rows,
        "baseline.info.kept": len(db["dfs"]),
        "baseline.info.candidates": db["run"].manifest["baseline"]["candidates"],
        "summary.proposed": len(proposals),
        "summary.accepted": statuses.count("accepted"),
        "summary.rejected": len(proposals) - statuses.count("accepted"),
        "summary.refused": statuses.count("rejected_guard") + statuses.count("rejected_duplicate"),
        "summary.vetoed": statuses.count("vetoed"),
        "summary.n_features": len(final.feature_set),
        "experiments.total": db["n_experiments"],
        "experiments.with_test_metrics": db["n_with_test"],
        "llm.n_calls": len(calls),
        "llm.tokens_in": sum(c["tokens_in"] for c in calls),
        "llm.tokens_out": sum(c["tokens_out"] for c in calls),
        "llm.cost_usd": sum(c["cost_usd"] for c in calls),
        "llm.n_fallback": sum(c["decision_mode"] == "fallback" for c in calls),
        "derived.test_positives": round(
            final.test_metrics["n_test"] * final.test_metrics["base_rate"]
        ),
        "acceptance.n_folds": len(proposals[0].gain["base_scores"]),
        "acceptance.min_gain": proposals[0].gain["rule"]["min_gain"],
        "acceptance.std_multiplier": proposals[0].gain["rule"]["std_multiplier"],
    }
    if key in simple:
        return float(simple[key])
    raise AssertionError(
        f"the report showed {key!r} and this test cannot check it against the database"
    )


async def load_db(run_id: str, cutoffs: list[dict[str, Any]]) -> dict[str, Any]:
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        exps = (await db.scalars(select(Experiment).where(Experiment.run_id == run_id))).all()
        feats = (
            await db.scalars(
                select(Feature).where(Feature.run_id == run_id).order_by(Feature.created_at)
            )
        ).all()
        spec = await db.get(TaskSpec, run.task_spec_id)
        version = await db.get(DataVersion, run.data_version_id)
    kind = lambda k: [e for e in exps if e.manifest.get("kind") == k]
    return {
        "run": run,
        "spec": spec,
        "version": version,
        "cutoffs": cutoffs,
        "final": kind("final")[0],
        "baseline": kind("baseline")[0],
        "champions": kind("champion"),
        "proposals": [f for f in feats if f.kind == "llm_sql"],
        "dfs": [f for f in feats if f.kind == "dfs"],
        "n_experiments": len(exps),
        "n_with_test": sum(e.test_metrics is not None for e in exps),
    }


def without_verbatim(text: str, verbatim: list[str]) -> str:
    """The text with every stored string (ids, dates, names, SQL, YAML) taken out: what is left
    must be words and the numbers the report logged."""
    for v in sorted(set(verbatim), key=len, reverse=True):
        for form in {v, " ".join(v.split()), " ".join(v.split()).replace("|", "\\|")}:
            text = text.replace(form, " ")
    return text


@pytest.mark.parametrize("accept_all", [False, True], ids=["default_rule", "all_kept"])
async def test_every_number_in_the_markdown_is_a_stored_number(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    accept_all: bool,
) -> None:
    run_id = await a_finished_run(client, project_id, connection, monkeypatch, accept_all)
    md_resp = await client.get(runs_url(project_id, f"/{run_id}/report"))
    assert md_resp.status_code == 200 and md_resp.headers["content-type"].startswith(
        "text/markdown"
    )
    md = md_resp.text
    data = (
        await client.get(runs_url(project_id, f"/{run_id}/report"), params={"format": "json"})
    ).json()
    records, log = data["records"], data["numbers"]

    for title in (
        "Summary",
        "Task",
        "Data",
        "Validation",
        "Features",
        "Leakage and safety",
        "Limitations",
        "Cost and reproducibility",
    ):
        assert f"\n## {title}\n" in md, title

    # 1. every numeral in the prose and tables is one the report logged
    verbatim = data["verbatim"]
    stored = set(records_text(records))
    foreign = [
        v
        for v in verbatim
        if v not in stored and not re.fullmatch(r"(top [\d.]+%|\d+% interval)", v)
    ]
    assert not foreign, f"stored text that is not in the records: {foreign}"
    shown = {e["shown"] for e in log}
    stray = [x for x in find_numbers(without_verbatim(md, verbatim)) if x not in shown]
    assert not stray, f"numbers in the report that no stored record produced: {stray}"

    # 2. each logged number is the stored value, formatted as the log says
    task_id = records["run"]["manifest"]["task"]["id"]
    preview = await client.post(
        f"{API}/projects/{project_id}/tasks/{task_id}/preview-labels",
        json={"data_version_id": records["data"]["id"]},
    )
    assert preview.status_code == 200, preview.text
    db = await load_db(run_id, preview.json()["cutoffs"])
    keys = set()
    for e in log:
        assert lookup(records, e["key"]) == pytest.approx(e["value"]), e
        stored = expected(e["key"], db)
        assert e["value"] == pytest.approx(stored), (e["key"], e["value"], stored)
        assert FORMATS[e["fmt"]](stored) == e["shown"], e
        keys.add(e["key"])
    assert len(keys) >= 60

    # the records themselves are the rows
    final = db["final"]
    assert records["final"]["test_metrics"] == final.test_metrics
    assert [f["gain"] for f in records["features"]] == [f.gain for f in db["proposals"]]
    assert records["run"]["manifest"] == db["run"].manifest
    assert db["n_with_test"] == 1  # the test rows were scored once

    # 3. stored text appears whole: the task, each kept feature's SQL
    assert db["spec"].yaml.strip() in md
    kept = [f for f in db["proposals"] if f.status == "accepted"]
    assert bool(kept) == accept_all and all(f.sql.strip() in md for f in kept)
    not_kept = [f for f in db["proposals"] if f.status != "accepted"]
    assert not_kept and all(f.name in md for f in not_kept)
    assert ("Champion path" in md) == accept_all
    assert {f.status for f in not_kept} == (
        {"rejected_guard"} if accept_all else {"rejected_gain", "rejected_guard"}
    )
    out_dir = os.environ.get("MLPILOT_REPORT_DIR")
    if out_dir:  # CI keeps the report of this run as an artifact
        name = "all-kept" if accept_all else "default-rule"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"report-{name}.md").write_text(md, encoding="utf-8")
        html = await client.get(
            runs_url(project_id, f"/{run_id}/report"), params={"format": "html"}
        )
        (Path(out_dir) / f"report-{name}.html").write_text(html.text, encoding="utf-8")
    t = final.test_metrics
    assert (
        f"are labelled 1 in {t['precision_at_10pct'] * 100:.1f}% of cases. That is "
        f"{t['lift_at_10pct']:.1f} times the base rate of {t['base_rate'] * 100:.1f}%, and together "
        f"they hold {t['recall_at_10pct'] * 100:.1f}% of all entities labelled 1"
    ) in md
    assert "feasibility check treats as comfortable" not in md  # 753 positives is not few
    print(
        f"\nreport: {len(md.splitlines())} lines, {len(log)} numbers shown ({len(keys)} distinct "
        f"records), {len(stray)} not traced to a record; test PR-AUC {t['pr_auc']:.4f}"
    )


def records_text(records: dict[str, Any]) -> list[str]:
    """Every string in the records: what ``Numbers.text`` may have shown verbatim."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(records)
    return out


async def test_the_html_report_opens_offline(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = await a_finished_run(client, project_id, connection, monkeypatch)
    resp = await client.get(runs_url(project_id, f"/{run_id}/report"), params={"format": "html"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    page = resp.text
    assert page.startswith("<!doctype html>") and "<svg" in page and "<h2" in page
    refs = re.findall(
        r"""\b(?:src|href|action|data|poster)\s*=\s*["']([^"']*)["']""", page, re.IGNORECASE
    )
    assert not [r for r in refs if "http" in r.lower() or r.startswith("//")], refs
    assert not re.search(r"<script|<link|<iframe|<img|@import|url\(", page, re.IGNORECASE)
    assert "http://" not in page and "https://" not in page
    # a download is the same file with a name
    dl = await client.get(
        runs_url(project_id, f"/{run_id}/report"), params={"format": "html", "download": "true"}
    )
    assert dl.headers["content-disposition"].startswith("attachment; filename=")
    print(
        f"\nhtml report: {len(page)} bytes, {len(refs)} src/href attributes, none external, no script"
    )


def test_a_run_without_a_final_model_or_proposals_says_what_is_missing() -> None:
    """The report keeps its sections and says 'not available' with the reason."""
    records = copy.deepcopy(SAMPLE)
    records["final"] = None
    records["features"] = []
    records["summary"].update(proposed=0, accepted=0, rejected=0, refused=0, vetoed=0)
    records["labels"] = {"available": False, "reason": "The labels could not be rebuilt now: gone"}
    records["schema"] = {"available": False, "reason": "The schema could not be read now: gone"}
    records["connection"] = {
        "available": False,
        "reason": "The task's connection no longer exists.",
    }
    records["run"].update(status="failed", error="boom")
    records["run"]["manifest"] = {
        "feasibility": {"status": "ok", "checks": [], "thresholds": {"warn_positives": 200}}
    }
    records["acceptance"] = None
    report = build_report(records)
    md = to_markdown(report)
    assert (
        "Answer: not available. The run did not reach its final model (status failed: boom)." in md
    )
    assert (
        "Label SQL and label balance: not available. The labels could not be rebuilt now: gone"
        in md
    )
    assert "Schema scan for leakage: not available." in md and "Database role: not available." in md
    assert "The language model proposed no features in this run." in md
    assert "The run ended with status failed (boom)" in md
    assert "<h2" in to_html(report)
    records["final"] = copy.deepcopy(SAMPLE["final"])
    md = to_markdown(build_report(records))
    assert "about 30 rows labelled 1 among 100, fewer than the 200 the feasibility" in md
    assert "<h2" in to_html(report)


def test_a_missing_record_is_an_error_not_a_blank() -> None:
    from ml.reports.numbers import MissingRecord

    records = copy.deepcopy(SAMPLE)
    del records["final"]["test_metrics"]["pr_auc"]
    with pytest.raises(MissingRecord, match="pr_auc"):
        build_report(records)


def metrics(**extra: float) -> dict[str, float]:
    """A complete set of ranking metrics, as the run stores them."""
    out = {"base_rate": 0.3, "pr_auc": 0.5}
    for cut in ("1pct", "5pct", "10pct"):
        out |= {f"precision_at_{cut}": 0.6, f"recall_at_{cut}": 0.2, f"lift_at_{cut}": 2.0}
    return out | extra


SAMPLE: dict[str, Any] = {
    "run": {
        "id": "r-1",
        "status": "completed",
        "error": None,
        "engine": "lightgbm",
        "seed": 42,
        "split_plan": {"val_from": "2024-07-01", "test_from": "2024-10-01", "folds": 3},
        "as_of": "2025-01-01T00:00:00+00:00",
        "finished_at": "2025-01-02T00:00:00+00:00",
        "budget": {},
        "budget_used": {},
        "manifest": {},
    },
    "task": {
        "name": "churn",
        "version": 1,
        "status": "confirmed",
        "confirmed_by": "me",
        "confirmed_at": "2025-01-01T00:00:00+00:00",
        "yaml": "name: churn\n",
        "words": "Predict churn.",
        "entity_table": "customers",
        "question": None,
        "decision_mode": None,
        "assumptions": [],
    },
    "data": {"id": "abc", "kind": "db_snapshot", "n_rows": 10, "tables": []},
    "baseline": {
        "experiment": {
            "val_metrics": metrics(),
            "engine": "lightgbm",
            "engine_version": "4",
        },
        "info": {"kept": 3, "candidates": 5},
        "features": [],
    },
    "final": {
        "model_name": "LGBMClassifier",
        "val_metrics": metrics(),
        "test_metrics": metrics(n_test=100.0),
        "feature_set": ["a", "b"],
        "manifest": {},
    },
    "features": [],
    "champions": [],
    "acceptance": None,
    "experiments": {"total": 2, "with_test_metrics": 1},
    "summary": {
        "proposed": 0,
        "accepted": 0,
        "rejected": 0,
        "refused": 0,
        "vetoed": 0,
        "n_features": 2,
    },
    "llm": {
        "calls": [],
        "n_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "n_fallback": 0,
    },
    "derived": {"test_positives": 30},
    "privacy": {"level": "schema_and_stats", "summary": "Schema and stats.", "never_send": []},
    "connection": {"available": True, "dialect": "duckdb", "can_write": False},
    "schema": {"available": True, "warnings": [], "tables": []},
    "labels": {"available": True, "sql": "SELECT 1", "cutoffs": []},
}


async def test_the_report_needs_a_baseline_and_a_known_run(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    early = await client.get(runs_url(project_id, f"/{run_id}/report"))
    assert early.status_code == 409 and "nothing to report" in early.text
    missing = await client.get(runs_url(project_id, "/does-not-exist/report"))
    assert missing.status_code == 404
    assert API  # the fixtures' base path is in use
