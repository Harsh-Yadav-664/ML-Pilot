"""The sections of the evidence report (#60), built from the records of one run.

Rules for this file:

* a figure is written with ``n.num(key, fmt)`` and nowhere else, so it is read from the records
  and listed in the report's number log;
* a stored string (id, date, name, stored message, SQL) is written with ``n.text(key)``;
* a fact that is not in the records is not stated. A section whose data is missing says so, and
  why, instead of leaving it out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.reports.numbers import MissingRecord, Numbers, lookup

TOP_CUTS = ["1pct", "5pct", "10pct"]
NOT_KEPT = {
    "rejected_gain": "Tested; the gain was not larger than the margin",
    "rejected_guard": "Refused by a check before it was tested",
    "rejected_duplicate": "Same as a feature the model already has",
    "vetoed": "Vetoed by a person before it was tested",
}
NOT_A = [
    (
        "It does not say why an entity is likely to be a positive. The features are patterns in "
        "past data, not causes."
    ),
    (
        "It ranks entities. Its scores were not checked for calibration, so a score is not a "
        "probability."
    ),
    (
        "It was trained on one snapshot and uses only what happened before each cutoff date. "
        "There is no drift monitoring: nothing here says it still holds after the data changes."
    ),
    (
        "It answers only the question in the task. A different horizon, label or entity needs "
        "a new run."
    ),
    "It has not been reviewed by a person unless the task section shows a confirmation.",
]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass
class Code:
    text: str
    lang: str = ""


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Chart:
    kind: str  # "split" or "gains"
    data: dict[str, Any]
    title: str


Block = str | Table | Code | Bullets | Chart


@dataclass
class Section:
    id: str
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Report:
    title: str
    meta: list[str]
    sections: list[Section]
    numbers: list[dict[str, Any]]
    verbatim: list[str]


def _get(records: dict[str, Any], path: str) -> Any:
    try:
        return lookup(records, path)
    except MissingRecord:
        return None


def _not_available(what: str, reason: str) -> str:
    return f"{what}: not available. {reason}"


def build_report(records: dict[str, Any]) -> Report:
    n = Numbers(records)
    sections = [
        _summary(records, n),
        _task(records, n),
        _data(records, n),
        _validation(records, n),
        _features(records, n),
        _safety(records, n),
        _limitations(records, n),
        _cost(records, n),
    ]
    meta = [
        f"Run {n.text('run.id')}",
        f"status {n.text('run.status')}",
        f"data version {n.text('data.id')}",
    ]
    if records["run"].get("finished_at"):
        meta.append(f"finished {n.text('run.finished_at')}")
    return Report(f"{n.text('task.name')}: run report", meta, sections, n.log(), n.verbatim)


# 1. Summary ---------------------------------------------------------------------------------


def _summary(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("summary", "Summary")
    s.blocks.append(
        f"Question: {n.text('task.question')}"
        if r["task"].get("question")
        else "Question: not recorded (the task was written by hand)."
    )
    final = r["final"]
    if final is None:
        error = f": {n.text('run.error')}" if r["run"].get("error") else ""
        s.blocks.append(
            "Answer: not available. The run did not reach its final model "
            f"(status {n.text('run.status')}{error})."
        )
        return s
    if final.get("test_metrics"):
        s.blocks.append(
            f"Answer: on the test period, which starts {n.text('run.split_plan.test_from')}, "
            f"the entities the model ranks in its {n.cut('10pct')} are labelled 1 in "
            f"{n.num('final.test_metrics.precision_at_10pct', 'pct1')}% of cases. That is "
            f"{n.num('final.test_metrics.lift_at_10pct', 'f1')} times the base rate of "
            f"{n.num('final.test_metrics.base_rate', 'pct1')}%, and together they hold "
            f"{n.num('final.test_metrics.recall_at_10pct', 'pct1')}% of all entities labelled 1 "
            f"({n.num('final.test_metrics.n_test', 'int')} rows scored). "
            "The test rows were scored once, after every decision was made."
        )
        s.blocks.append(
            Table(
                ["Metric", "Test", "Final model, validation", "Baseline, validation"],
                _metric_rows(n, r),
            )
        )
    else:
        why = final["manifest"].get("test_error")
        s.blocks.append(
            "Answer: not available. The final model could not be scored on the test rows"
            + (f": {n.text('final.manifest.test_error')}" if why else "")
            + f". Validation PR-AUC of the final model is {n.num('final.val_metrics.pr_auc', 'f4')} "
            f"against a base rate of {n.num('final.val_metrics.base_rate', 'pct1')}%."
        )
    s.blocks.append(
        f"Model: {n.text('final.model_name')}, {n.num('summary.n_features', 'int')} features: "
        f"the {n.num('baseline.info.kept', 'int')} automatic baseline features and "
        f"{n.num('summary.accepted', 'int')} feature(s) proposed by the language model and kept "
        f"({n.num('summary.proposed', 'int')} proposed in all). Seed {n.num('run.seed', 'int')}."
    )
    s.blocks.append(
        "The validation columns are the numbers the run used to pick features and to stop the "
        "model early, so they are higher than what to expect on new data. Only the test column "
        "estimates that."
    )
    return s


def _metric_rows(n: Numbers, r: dict[str, Any]) -> list[list[str]]:
    roots = ("final.test_metrics", "final.val_metrics", "baseline.experiment.val_metrics")

    def row(label: str, key: str, fmt: str) -> list[str]:
        # a column is "n/a" only when that whole model has no such metrics; a missing key in
        # metrics that exist is an error
        return [label] + [
            n.num(f"{c}.{key}", fmt) if _get(r, c) is not None else "n/a" for c in roots
        ]

    rows = [row("Base rate (%)", "base_rate", "pct1"), row("PR-AUC", "pr_auc", "f4")]
    for key in TOP_CUTS:
        label = n.cut(key)
        rows.append(row(f"Precision (%), {label}", f"precision_at_{key}", "pct1"))
        rows.append(row(f"Recall (%), {label}", f"recall_at_{key}", "pct1"))
        rows.append(row(f"Lift, {label}", f"lift_at_{key}", "f2"))
    return rows


# 2. Task ------------------------------------------------------------------------------------


def _task(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("task", "Task")
    t = r["task"]
    confirmed = (
        f", confirmed by {n.text('task.confirmed_by')} on {n.text('task.confirmed_at')}"
        if t.get("confirmed_at")
        else ""
    )
    s.blocks.append(
        f"Version {n.num('task.version', 'int')}, status {n.text('task.status')}{confirmed}."
    )
    s.blocks.append(n.text("task.words"))
    s.blocks.append("The task as stored:")
    s.blocks.append(Code(n.text("task.yaml"), "yaml"))
    if t.get("question"):
        s.blocks.append(
            f"It was drafted from the question by the {n.text('task.decision_mode')} path."
        )
    if t.get("assumptions"):
        s.blocks.append("Assumptions made when the question was turned into a task:")
        s.blocks.append(
            Bullets([n.text(f"task.assumptions[{i}]") for i in range(len(t["assumptions"]))])
        )
    labels = r["labels"]
    if not labels["available"]:
        s.blocks.append(_not_available("Label SQL and label balance", labels["reason"]))
    else:
        s.blocks.append("The label query (it only reads, and runs on the snapshot):")
        s.blocks.append(Code(n.text("labels.sql"), "sql"))
        rows = []
        for i, c in enumerate(labels["cutoffs"]):
            p = f"labels.cutoffs[{i}]"
            rows.append(
                [
                    n.text(f"{p}.cutoff"),
                    n.num(f"{p}.eligible", "int"),
                    n.num(f"{p}.positives", "int") if c["positives"] is not None else "n/a",
                    n.num(f"{p}.base_rate", "pct1") if c["base_rate"] is not None else "n/a",
                ]
            )
        s.blocks.append("Label balance per cutoff:")
        s.blocks.append(Table(["Cutoff", "Entities", "Labelled 1", "Base rate (%)"], rows))
    feas = r["run"]["manifest"].get("feasibility")
    if feas:
        s.blocks.append(
            f"Feasibility checks before the run: {n.text('run.manifest.feasibility.status')}."
        )
        s.blocks.append(
            Table(
                ["Check", "Result", "What it found"],
                [
                    [
                        n.text(f"run.manifest.feasibility.checks[{i}].code"),
                        n.text(f"run.manifest.feasibility.checks[{i}].status"),
                        n.text(f"run.manifest.feasibility.checks[{i}].message"),
                    ]
                    for i in range(len(feas["checks"]))
                ],
            )
        )
    override = r["run"]["manifest"].get("override") or {}
    if override.get("used"):
        reason = n.text("run.manifest.override.reason") if override.get("reason") else "none given"
        s.blocks.append(f"The run was started over a blocking check. Reason: {reason}.")
    return s


# 3. Data ------------------------------------------------------------------------------------


def _data(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("data", "Data")
    d = r["data"]
    as_of = f", read as of {n.text('run.as_of')}" if r["run"].get("as_of") else ""
    size = f"; {n.num('data.n_rows', 'int')} rows" if d.get("n_rows") is not None else ""
    s.blocks.append(f"Data version {n.text('data.id')}: a {n.text('data.kind')}{as_of}{size}.")
    if d["tables"]:
        s.blocks.append(
            Table(
                ["Table", "Rows", "Latest event time", "Rows left out (no event time)"],
                [
                    [
                        n.text(f"data.tables[{i}].name"),
                        n.num(f"data.tables[{i}].rows", "int")
                        if t.get("rows") is not None
                        else "n/a",
                        n.text(f"data.tables[{i}].max_event_time")
                        if t.get("max_event_time")
                        else "n/a",
                        n.num(f"data.tables[{i}].null_time_rows", "int")
                        if t.get("null_time_rows") is not None
                        else "n/a",
                    ]
                    for i, t in enumerate(d["tables"])
                ],
            )
        )
    skipped = r["baseline"]["info"].get("skipped_tables") or []
    if skipped:
        s.blocks.append("Tables the baseline did not use:")
        s.blocks.append(
            Bullets(
                [
                    f"{n.text(f'baseline.info.skipped_tables[{i}].table')}: "
                    f"{n.text(f'baseline.info.skipped_tables[{i}].reason')}"
                    for i in range(len(skipped))
                ]
            )
        )
    schema = r["schema"]
    if schema["available"]:
        static = [i for i, t in enumerate(schema["tables"]) if t.get("is_static")]
        if static:
            names = ", ".join(n.text(f"schema.tables[{i}].key") for i in static)
            s.blocks.append(
                f"Tables treated as static (no event time, so every row counts as known at every cutoff): {names}."
            )
    else:
        s.blocks.append(_not_available("Static-table assumptions", schema["reason"]))
    p = r["privacy"]
    never = (
        f" Columns excluded from every prompt: {', '.join(n.text(f'privacy.never_send[{i}]') for i in range(len(p['never_send'])))}."
        if p["never_send"]
        else ""
    )
    s.blocks.append(
        f"Privacy: the project's setting when this report was made is {n.text('privacy.level')}: "
        f"{n.text('privacy.summary')}{never}"
    )
    return s


# 4. Validation ------------------------------------------------------------------------------


def _validation(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("validation", "Validation")
    s.blocks.append(
        f"The data is split by date, never at random. Cutoffs before {n.text('run.split_plan.val_from')} "
        f"train the model, cutoffs from then until {n.text('run.split_plan.test_from')} validate it, "
        f"and cutoffs from {n.text('run.split_plan.test_from')} on are the test. A row whose label "
        "window reaches into the next period is left out of its own."
    )
    labels = r["labels"]
    plan = r["run"]["split_plan"]
    if labels["available"]:
        s.blocks.append(
            Chart(
                "split",
                {
                    "cutoffs": [c["cutoff"] for c in labels["cutoffs"]],
                    "val_from": plan["val_from"],
                    "test_from": plan["test_from"],
                },
                "Cutoffs by period",
            )
        )
    if r["acceptance"] is None:
        s.blocks.append("Acceptance rule: no feature reached the gain test in this run.")
    else:
        s.blocks.append(
            "A feature is kept only if its paired gain in PR-AUC over the model without it, "
            f"measured on {n.num('acceptance.n_folds', 'int')} time-ordered folds of the training "
            "rows, is larger than the margin. The margin is the larger of "
            f"{n.num('acceptance.std_multiplier', 'f1')} standard deviation(s) of the gain and "
            f"{n.num('acceptance.min_gain', 'f4')}. Code applies the rule; the language model does "
            "not decide."
        )
    s.blocks.append(
        f"Experiments of this run that have test metrics: {n.num('experiments.with_test_metrics', 'int')} "
        f"of {n.num('experiments.total', 'int')}. Only the final model is scored on the test rows, "
        "and once. No feature and no setting was chosen with them."
    )
    return s


# 5. Features --------------------------------------------------------------------------------


def _features(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("features", "Features")
    base = r["baseline"]
    s.blocks.append(
        f"Baseline: {n.num('baseline.info.kept', 'int')} automatic features kept out of "
        f"{n.num('baseline.info.candidates', 'int')} candidates, built without a language model."
    )
    top = base["features"][:10]
    if top:
        s.blocks.append("Largest baseline features, by share of the model's gain:")
        s.blocks.append(
            Table(
                ["Feature", "What it is", "Share of gain (%)"],
                [
                    [
                        n.text(f"baseline.features[{i}].name"),
                        n.text(f"baseline.features[{i}].description"),
                        n.num(f"baseline.features[{i}].importance", "pct1"),
                    ]
                    for i in range(len(top))
                ],
            )
        )
    s.blocks.append("Feature importance by SHAP values: not available yet.")
    proposals = r["features"]
    if not proposals:
        s.blocks.append("The language model proposed no features in this run.")
        return s
    accepted = [i for i, f in enumerate(proposals) if f["status"] == "accepted"]
    if accepted:
        s.blocks.append(f"Features kept ({n.num('summary.accepted', 'int')}):")
    for i in accepted:
        p = f"features[{i}]"
        g = proposals[i]["gain"]
        s.blocks.append(f"{n.text(p + '.name')}: {n.text(p + '.description')}")
        s.blocks.append(Code(n.text(p + ".sql"), "sql"))
        base_scores = ", ".join(
            n.num(f"{p}.gain.base_scores[{k}]", "f4") for k in range(len(g["base_scores"]))
        )
        with_scores = ", ".join(
            n.num(f"{p}.gain.candidate_scores[{k}]", "f4")
            for k in range(len(g["candidate_scores"]))
        )
        s.blocks.append(
            f"Paired gain in PR-AUC {n.sign(p + '.gain.mean_gain', 'f4')}, {n.interval('ci95')} "
            f"{n.sign(p + '.gain.ci95[0]', 'f4')} to {n.sign(p + '.gain.ci95[1]', 'f4')}, margin "
            f"{n.num(p + '.gain.margin', 'f4')}. Fold scores without the feature: {base_scores}. "
            f"With it: {with_scores}."
        )
        s.blocks.append(
            "Checks: it passed every check before the gain test (stored status "
            f"{n.text(p + '.guard_results.status')})."
        )
    rows = []
    for i, f in enumerate(proposals):
        if f["gain"] is None:
            continue
        p = f"features[{i}]"
        rows.append(
            {
                "name": f["name"],
                "mean": f["gain"]["mean_gain"],
                "lo": f["gain"]["ci95"][0],
                "hi": f["gain"]["ci95"][1],
                "margin": f["gain"]["margin"],
                "accepted": f["status"] == "accepted",
                "label": n.sign(p + ".gain.mean_gain", "f4"),
            }
        )
    if rows:
        s.blocks.append(
            Chart(
                "gains",
                {"rows": rows},
                f"Paired gain of each tested feature, with its {n.interval('ci95')}",
            )
        )
    if r["summary"]["rejected"]:
        s.blocks.append(f"Not kept ({n.num('summary.rejected', 'int')}), by reason:")
    for status, label in NOT_KEPT.items():
        group = [i for i, f in enumerate(proposals) if f["status"] == status]
        if not group:
            continue
        items = []
        for i in group:
            p = f"features[{i}]"
            line = n.text(p + ".name")
            f = proposals[i]
            if f["gain"] is not None:
                line += f": gain {n.sign(p + '.gain.mean_gain', 'f4')}, margin {n.num(p + '.gain.margin', 'f4')}"
            elif (f["guard_results"] or {}).get("reasons"):
                line += f": {n.text(p + '.guard_results.reasons[0]')}"
            items.append(line)
        s.blocks.append(f"{label}:")
        s.blocks.append(Bullets(items))
    if r["champions"]:
        s.blocks.append("Champion path (validation PR-AUC after each kept feature):")
        s.blocks.append(
            Table(
                ["Round", "Added", "Validation PR-AUC"],
                [
                    [
                        n.num(f"champions[{i}].round", "int"),
                        n.text(f"champions[{i}].feature"),
                        n.num(f"champions[{i}].val_pr_auc", "f4"),
                    ]
                    for i in range(len(r["champions"]))
                ],
            )
        )
    return s


# 6. Leakage and safety ----------------------------------------------------------------------


def _safety(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("safety", "Leakage and safety")
    conn = r["connection"]
    if not conn["available"]:
        s.blocks.append(_not_available("Database role", conn["reason"]))
    elif conn["can_write"] is False:
        s.blocks.append("Database role: checked, and it cannot write.")
    elif conn["can_write"] is True:
        s.blocks.append(
            "Database role: it CAN write. MLPilot only sends single SELECT statements in a "
            "read-only transaction, but the role itself should be restricted."
        )
    else:
        s.blocks.append("Database role: whether it can write has not been checked.")
    s.blocks.append(
        "Every query goes through the SQL guard: one SELECT, parsed, with a time and a row "
        "limit. A feature query must use only rows dated before the cutoff of the row it feeds."
    )
    s.blocks.append(
        f"Proposed features: {n.num('summary.proposed', 'int')}. Refused by a check before "
        f"testing: {n.num('summary.refused', 'int')}. Vetoed by a person: "
        f"{n.num('summary.vetoed', 'int')}."
    )
    schema = r["schema"]
    if not schema["available"]:
        s.blocks.append(_not_available("Schema scan for leakage", schema["reason"]))
    else:
        flagged = [i for i, t in enumerate(schema["tables"]) if t.get("leakage_hint")]
        mutable = [(i, j) for i, t in enumerate(schema["tables"]) for j in range(len(t["mutable"]))]
        if flagged:
            s.blocks.append(
                "Tables that look like the current state, not a history (possible leakage):"
            )
            s.blocks.append(
                Bullets(
                    [
                        f"{n.text(f'schema.tables[{i}].key')}: {n.text(f'schema.tables[{i}].leakage_hint')}"
                        for i in flagged
                    ]
                )
            )
        if mutable:
            s.blocks.append(
                "Columns that may be overwritten after their row's event time (a feature built on one can see the future):"
            )
            s.blocks.append(
                Bullets(
                    [
                        f"{n.text(f'schema.tables[{i}].key')}.{n.text(f'schema.tables[{i}].mutable[{j}].name')} "
                        f"(found by {n.text(f'schema.tables[{i}].mutable[{j}].source')})"
                        for i, j in mutable
                    ]
                )
            )
        if not flagged and not mutable:
            s.blocks.append(
                "The schema scan found no table that looks like a current-state snapshot and no mutable column."
            )
        for i in range(len(schema["warnings"])):
            s.blocks.append(f"Schema warning: {n.text(f'schema.warnings[{i}]')}")
    s.blocks.append(
        "The leakage canaries (planted columns the checks must catch) run in the project's test "
        "suite, not per run, so this report has no canary result for this run."
    )
    return s


# 7. Limitations -----------------------------------------------------------------------------


def _limitations(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("limitations", "Limitations")
    facts: list[str] = []
    if r["run"]["status"] != "completed":
        why = f" ({n.text('run.error')})" if r["run"].get("error") else ""
        facts.append(
            f"The run ended with status {n.text('run.status')}{why}; the model is the best one found before it stopped."
        )
    final = r["final"]
    warn_at = _get(r, "run.manifest.feasibility.thresholds.warn_positives")
    positives = r["derived"].get("test_positives")
    if (
        final
        and final.get("test_metrics")
        and positives is not None
        and warn_at is not None
        and positives < warn_at
    ):
        facts.append(
            f"The test period has about {n.num('derived.test_positives', 'int')} rows labelled 1 "
            f"among {n.num('final.test_metrics.n_test', 'int')}, fewer than the "
            f"{n.num('run.manifest.feasibility.thresholds.warn_positives', 'int')} the feasibility "
            "check treats as comfortable. The test numbers are noisy, and no interval is given for them."
        )
    labels = r["labels"]
    rated = (
        [(i, c["base_rate"]) for i, c in enumerate(labels["cutoffs"]) if c.get("base_rate")]
        if labels["available"]
        else []
    )
    if rated:
        lo = min(rated, key=lambda x: x[1])
        hi = max(rated, key=lambda x: x[1])
        if hi[1] >= 2 * lo[1]:
            facts.append(
                f"The base rate moves between cutoffs, from {n.num(f'labels.cutoffs[{lo[0]}].base_rate', 'pct1')}% "
                f"to {n.num(f'labels.cutoffs[{hi[0]}].base_rate', 'pct1')}%. A model that ranks well "
                "at one rate can rank differently at another."
            )
    sampling = (r["run"]["manifest"].get("run_loop") or {}).get("feature_engine", {}).get(
        "sampling"
    ) or {}
    if sampling.get("sampled"):
        facts.append(
            "Feature values were computed on a sample of the entities, not on all of them."
        )
    if r["summary"]["proposed"] > 0:
        facts.append(
            f"{n.num('summary.proposed', 'int')} feature(s) were tested on the same few folds and "
            f"{n.num('summary.accepted', 'int')} kept. With that many tries, a feature with no real "
            "effect is sometimes kept by chance."
        )
    if r["data"]["kind"] != "db_snapshot":
        facts.append("The data was read live, so the run cannot be reproduced exactly.")
    s.blocks.append("From the facts of this run:")
    s.blocks.append(Bullets(facts) if facts else "Nothing beyond the list below.")
    s.blocks.append("What this model is not:")
    s.blocks.append(Bullets(NOT_A))
    return s


# 8. Cost and reproducibility ----------------------------------------------------------------


def _cost(r: dict[str, Any], n: Numbers) -> Section:
    s = Section("cost", "Cost and reproducibility")
    calls = r["llm"]["calls"]
    if calls:
        s.blocks.append(
            f"Language-model calls for the proposed features: {n.num('llm.n_calls', 'int')}, "
            f"{n.num('llm.tokens_in', 'int')} tokens in and {n.num('llm.tokens_out', 'int')} out, "
            f"cost ${n.num('llm.cost_usd', 'usd')}. Answered by the offline fallback instead of a "
            f"model: {n.num('llm.n_fallback', 'int')}."
        )
        s.blocks.append(
            Table(
                [
                    "Provider",
                    "Model",
                    "Mode",
                    "Tokens in",
                    "Tokens out",
                    "Cost ($)",
                    "Prompt log id",
                ],
                [
                    [
                        n.text(f"llm.calls[{i}].provider"),
                        n.text(f"llm.calls[{i}].model"),
                        n.text(f"llm.calls[{i}].decision_mode"),
                        n.num(f"llm.calls[{i}].tokens_in", "int"),
                        n.num(f"llm.calls[{i}].tokens_out", "int"),
                        n.num(f"llm.calls[{i}].cost_usd", "usd"),
                        n.text(f"llm.calls[{i}].log_id"),
                    ]
                    for i in range(len(calls))
                ],
            )
        )
        s.blocks.append(
            "Every prompt and answer is stored in the project's prompt log under the id in the last column."
        )
    else:
        s.blocks.append("No language-model call is recorded for the features of this run.")
    run = r["run"]
    if run["budget_used"].get("cost_usd") is not None:
        limit = (
            f"${n.num('run.budget.max_cost_usd', 'usd')}"
            if run["budget"].get("max_cost_usd") is not None
            else "no cost limit"
        )
        stopped = (
            f"; stopped by: {n.text('run.budget_used.stopped')}"
            if run["budget_used"].get("stopped")
            else ""
        )
        s.blocks.append(
            f"Budget used: ${n.num('run.budget_used.cost_usd', 'usd')} against {limit}{stopped}."
        )
    engine = r["baseline"]["experiment"]
    how = f"seed {n.num('run.seed', 'int')}"
    if engine.get("engine"):
        how += f", {n.text('baseline.experiment.engine')} {n.text('baseline.experiment.engine_version')}"
    s.blocks.append("To reproduce this run you need:")
    s.blocks.append(
        Bullets(
            [
                f"data version {n.text('data.id')}",
                f"task version {n.num('task.version', 'int')} (the YAML above)",
                (
                    f"split {n.text('run.split_plan.val_from')} / {n.text('run.split_plan.test_from')}, "
                    f"{n.num('run.split_plan.folds', 'int')} folds"
                ),
                how,
                "the feature SQL listed above; the language model's answers are in the prompt log",
            ]
        )
    )
    return s
