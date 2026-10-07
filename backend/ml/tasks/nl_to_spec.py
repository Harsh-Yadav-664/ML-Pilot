"""A business question in, a validated task spec out (#53).

The LLM proposes; deterministic code validates; the human confirms. ``draft_spec``:

1. builds the prompt through the context builder (the question, the schema graph and statistics
   at the project's privacy level, the spec's JSON Schema and three worked examples),
2. asks for ``{spec, assumptions, clarifying_question}``,
3. checks the spec with the task spec validator (#49) and, if it has errors, asks once more with
   the errors written out,
4. returns a ``SpecDraft``: a spec ready to preview and confirm, or a clarifying question, or the
   problems that remain. Nothing is saved or run here.

A vague question ("predict customers", no horizon, no event) gets a clarifying question, never a
guess. When no LLM answered (the offline stub, or every provider failed) the rule-based drafter
``rule_draft`` handles the questions it recognises and is recorded as ``decision_mode:
fallback``; it asks a clarifying question for the rest.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

import yaml

from ai.context_builder import ContextBuilder, PromptBuilder
from ai.context_inputs import dataset_from_graph
from ai.gateway import AIGateway
from ai.router import TaskType
from ml.data.profiling.db_stats import TableStats
from ml.data.schema_graph import Edge, SchemaGraph, Table, type_kind
from ml.tasks import prompts
from ml.tasks.spec import (
    SpecError,
    SpecIssue,
    TaskSpec,
    add_duration,
    from_dict,
    json_schema,
    parse_duration,
    to_yaml,
    validate_against,
)

Status = Literal["spec", "clarify", "invalid"]
DecisionMode = Literal["llm", "fallback"]
MONTHS_OF_CUTOFFS = 18
CLARIFY_EXAMPLE = 'For example: "Which customers will stop ordering in the next 30 days?"'


@dataclass
class SpecDraft:
    status: Status
    question: str
    decision_mode: DecisionMode
    spec: TaskSpec | None = None
    assumptions: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    issues: list[SpecIssue] = field(default_factory=list)
    repaired: bool = False
    llm: list[dict[str, Any]] = field(default_factory=list)  # one meta dict per LLM call

    @property
    def yaml(self) -> str | None:
        return to_yaml(self.spec) if self.spec is not None else None

    @property
    def errors(self) -> list[SpecIssue]:
        return [i for i in self.issues if i.severity == "error"]


def data_end(graph: SchemaGraph, stats: dict[str, TableStats] | None, now: datetime) -> datetime:
    """Where the data ends: the latest value of any event-time column, never later than ``now``.

    A database nobody has written to for a year would otherwise get cutoffs whose label windows
    are empty, and every entity would look like a churner."""
    latest: datetime | None = None
    for table in graph.tables:
        if not table.time_column or table.is_static or not stats or table.key not in stats:
            continue
        column = next((c for c in stats[table.key].columns if c.name == table.time_column), None)
        if column is None or not column.time_max:
            continue
        try:
            value = datetime.fromisoformat(str(column.time_max))
        except ValueError:
            continue
        value = value if value.tzinfo else value.replace(tzinfo=UTC)
        latest = value if latest is None or value > latest else latest
    return min(latest, now) if latest else now


# -- the structured answer the LLM gives -----------------------------------------------------


def answer_schema() -> dict[str, Any]:
    """JSON Schema of the LLM's answer: the task spec's own schema plus assumptions and a question."""
    spec_schema = json_schema()
    defs = spec_schema.pop("$defs", {})
    spec_schema.pop("title", None)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "spec": spec_schema,
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Plain sentences, one per judgement call",
            },
            "clarifying_question": {
                "type": ["string", "null"],
                "description": "Set instead of a spec when the question is too vague",
            },
        },
        "required": ["assumptions"],
    }
    if defs:
        schema["$defs"] = defs
    return schema


def _compose(
    builder: ContextBuilder,
    question: str,
    graph: SchemaGraph,
    stats: dict[str, TableStats] | None,
    as_of: datetime,
    extra: str | None = None,
) -> PromptBuilder:
    examples: list[dict[str, Any]] = [
        {
            "question": e["question"],
            "assumptions": e.get("assumptions", []),
            "spec": yaml.safe_load(e["spec"]) if e.get("spec") else None,
            "clarifying_question": e.get("clarifying_question"),
        }
        for e in prompts.worked_examples()
    ]
    b = (
        builder.prompt("tasks.draft_spec", prompts.system_prompt())
        .text("Question", question)
        .dataset(dataset_from_graph(graph, stats))
        .facts("Data ends", {"as_of": as_of.date().isoformat()})
        .facts("Task spec JSON Schema", json_schema())
        .facts("Worked examples", examples)
    )
    if extra:
        b = b.text("Your previous answer had problems", extra)
    return b


# -- the pipeline -----------------------------------------------------------------------------


async def draft_spec(
    question: str,
    graph: SchemaGraph,
    *,
    gateway: AIGateway,
    builder: ContextBuilder,
    as_of: datetime,
    stats: dict[str, TableStats] | None = None,
) -> SpecDraft:
    """Draft a task spec for ``question``. Raises ``ValueError`` for an empty question."""
    question = question.strip()
    if not question:
        raise ValueError("Ask a question, such as: " + CLARIFY_EXAMPLE)
    schema = answer_schema()
    prompt = _compose(builder, question, graph, stats, as_of).build()
    result = await gateway.complete_structured_result(TaskType.SPEC, prompt, schema)
    if result.decision_mode == "fallback":
        # No real LLM answered: the offline stub, or every provider failed. Say so, and use rules.
        draft = rule_draft(question, graph, as_of)
        draft.llm = [result.meta()]
        return draft
    draft = _interpret(question, result.structured, graph, as_of)
    draft.llm = [result.meta()]
    if draft.status == "invalid":
        problems = _problems_text(draft)
        again = await gateway.complete_structured_result(
            TaskType.SPEC,
            _compose(builder, question, graph, stats, as_of, extra=problems).build(),
            schema,
        )
        second = _interpret(question, again.structured, graph, as_of)
        second.repaired = True
        second.llm = [*draft.llm, again.meta()]
        return second
    return draft


def _interpret(
    question: str, data: dict[str, Any], graph: SchemaGraph, as_of: datetime
) -> SpecDraft:
    assumptions = [str(a) for a in data.get("assumptions") or [] if str(a).strip()]
    ask = data.get("clarifying_question")
    raw = data.get("spec")
    if isinstance(ask, str) and ask.strip() and not raw:
        return SpecDraft("clarify", question, "llm", None, assumptions, ask.strip())
    try:
        spec = from_dict(raw)
    except SpecError as e:
        return SpecDraft("invalid", question, "llm", None, assumptions, None, list(e.issues))
    issues = validate_against(spec, graph, as_of)
    status: Status = "invalid" if any(i.severity == "error" for i in issues) else "spec"
    return SpecDraft(status, question, "llm", spec, assumptions, None, issues)


def _problems_text(draft: SpecDraft) -> str:
    lines = [f"- {i.path or 'spec'}: {i.message}" for i in draft.errors]
    spec = json.dumps(draft.spec.model_dump(mode="json", exclude_none=True)) if draft.spec else ""
    return (
        "The spec you gave was checked against the database and has these errors:\n"
        + "\n".join(lines)
        + (f"\nYour spec was: {spec}" if spec else "")
        + "\nGive a corrected answer in the same format. Do not ask a question unless you must."
    )


# -- plain words ------------------------------------------------------------------------------


def _singular(name: str) -> str:
    n = name.lower()
    if n.endswith("ies"):
        return n[:-3] + "y"
    if n.endswith("s") and not n.endswith("ss"):
        return n[:-1]
    return n


def _words(text: str) -> str:
    return text.replace("_", " ")


def _days(duration: str) -> str:
    """'30d' as '30 days' for the card; other forms ('1 month') are already words."""
    m = re.fullmatch(r"(\d+)\s*d", duration.strip(), re.IGNORECASE)
    return f"{m.group(1)} days" if m else _words(duration)


def describe_spec(spec: TaskSpec) -> str:
    """The spec in plain sentences, for the card the user confirms."""
    entity = _singular(spec.entity.table)
    ex = spec.target.expression
    horizon = _days(spec.horizon)
    if ex is None:
        label = "the label comes from a custom SQL query (flagged in the report)"
    else:
        what = _words(ex.table) + (f" ({ex.where})" if ex.where else "")
        if spec.target.type == "regression":
            part = f"{ex.agg} of {ex.column or ex.table} over {what}"
            label = f"The value to predict is the {part} in the {horizon} after the cutoff."
        else:
            compare = (ex.compare or "").replace(" ", "")
            count = f"the number of {what} rows" if ex.agg == "count" else f"the {ex.agg} of {what}"
            if ex.agg == "count" and compare == "=0":
                label = (
                    f"A {entity} is labelled 1 (what we predict) if there is no row in {what} "
                    f"in the {horizon} after the cutoff."
                )
            else:
                label = (
                    f"A {entity} is labelled 1 (what we predict) if {count} in the {horizon} "
                    f"after the cutoff is {ex.compare}."
                )
    rules = []
    for rule in spec.eligibility:
        if isinstance(rule, str):
            rules.append(rule.replace(":cutoff", "the cutoff"))
        else:
            test = rule.exists or rule.not_exists
            assert test is not None
            has = "has" if rule.exists else "has no"
            rules.append(
                f"{has} {_words(test.table)} rows"
                + (f" where {test.where.replace(':cutoff', 'the cutoff')}" if test.where else "")
            )
    scored = (
        f"Scored at each cutoff: {entity}s for which " + "; ".join(rules) + "."
        if rules
        else f"Scored at each cutoff: every {entity}."
    )
    n = len(_cutoffs(spec))
    cutoffs = (
        f"Cutoffs: every {_words(spec.cutoffs.every)} from {spec.cutoffs.start} to "
        f"{spec.cutoffs.end} ({n} cutoffs). Validation from {spec.split.val_from}, test from "
        f"{spec.split.test_from}. Judged by {spec.metric_name}."
    )
    return f"{label} {scored} {cutoffs}"


def _cutoffs(spec: TaskSpec) -> list[date]:
    from ml.tasks.spec import cutoff_dates

    return cutoff_dates(spec)


# -- the rule-based drafter (offline, deterministic) -------------------------------------------

_HORIZON = re.compile(r"\b(\d+)\s*(days?|d|weeks?|wks?|w|months?)\b", re.IGNORECASE)
_NEXT = re.compile(
    r"\b(?:next|coming|within a|in a|in the next)\s+(week|month|quarter)\b", re.IGNORECASE
)
_ENTITY_WORDS = {
    "customer": ("customer", "client", "shopper", "buyer"),
    "user": ("user", "member", "subscriber", "account"),
}


@dataclass(frozen=True)
class _Outcome:
    kind: Literal["none", "event", "amount"]
    pattern: re.Pattern[str]
    tables: tuple[str, ...]  # words the event table's name may contain
    compare: str
    name: str
    needs_history: bool = False  # only score entities that already have a row in the table


_OUTCOMES = [
    _Outcome(
        "none",
        re.compile(
            r"stop (?:ordering|buying|purchasing)|churn|lapse|no longer (?:order|buy)|"
            r"won'?t (?:order|buy)|not (?:order|buy)|become inactive|go inactive",
            re.IGNORECASE,
        ),
        ("order", "purchase", "transaction"),
        "= 0",
        "no_{table}",
        True,
    ),
    _Outcome(
        "event",
        re.compile(
            r"refund|return(?:s|ed)? (?:an? )?(?:order|item)|ask for (?:a )?refund", re.IGNORECASE
        ),
        ("refund",),
        ">= 1",
        "{table}",
    ),
    _Outcome(
        "event",
        re.compile(
            r"complain|complaint|support ticket|open (?:a )?ticket|contact support", re.IGNORECASE
        ),
        ("ticket", "complaint"),
        ">= 1",
        "{table}",
    ),
    _Outcome(
        "event",
        re.compile(
            r"order again|buy again|repeat (?:purchase|order)|reorder|place (?:an? )?order",
            re.IGNORECASE,
        ),
        ("order", "purchase", "transaction"),
        ">= 1",
        "{table}",
        True,
    ),
    _Outcome(
        "amount",
        re.compile(r"how much|spend|revenue|total value|lifetime value|\bltv\b", re.IGNORECASE),
        ("order", "purchase", "transaction"),
        "",
        "spend",
    ),
]
_AMOUNT_COLUMNS = ("total", "amount", "price", "revenue", "value", "spend")


def _clarify(question: str, why: str, assumptions: list[str] | None = None) -> SpecDraft:
    return SpecDraft(
        "clarify",
        question,
        "fallback",
        None,
        assumptions or [],
        f"{why} {CLARIFY_EXAMPLE}",
    )


def _horizon(question: str) -> str | None:
    m = _HORIZON.search(question)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if n == 0:
            return None
        if unit.startswith("d"):
            return f"{n}d"
        if unit.startswith("w"):
            return f"{n * 7}d"
        return f"{n * 30}d"
    nxt = _NEXT.search(question)
    if nxt:
        return {"week": "7d", "month": "30d", "quarter": "90d"}[nxt.group(1).lower()]
    return None


def _entity(question: str, graph: SchemaGraph) -> tuple[Table | None, str | None]:
    """The table the question is about: named in it, or the one most tables point at."""
    q = question.lower()
    incoming: dict[str, int] = {}
    for e in graph.edges:
        incoming[e.to_table] = incoming.get(e.to_table, 0) + 1
    candidates = [t for t in graph.tables if t.primary_key and incoming.get(t.key)]
    for t in candidates:
        words = {t.name.lower(), _singular(t.name)}
        if any(re.search(rf"\b{re.escape(w)}s?\b", q) for w in words):
            return t, None
    for table_word, synonyms in _ENTITY_WORDS.items():
        if any(re.search(rf"\b{w}s?\b", q) for w in synonyms):
            for t in candidates:
                if table_word in t.name.lower() or any(s in t.name.lower() for s in synonyms):
                    return t, None
    if candidates:
        top = max(incoming[t.key] for t in candidates)
        best = [t for t in candidates if incoming[t.key] == top]
        if len(best) == 1:
            return (
                best[0],
                f"The question does not name who is predicted about; {best[0].name} is the table most others point at.",
            )
    return None, None


def _event_table(
    graph: SchemaGraph, entity: Table, words: tuple[str, ...]
) -> tuple[Table, Edge] | None:
    for t in graph.tables:
        if t.key == entity.key or not t.time_column or t.is_static:
            continue
        if not any(w in t.name.lower() for w in words):
            continue
        edges = [e for e in graph.edges if e.from_table == t.key and e.to_table == entity.key]
        if edges:
            return t, edges[0]
    return None


def _created_at(entity: Table) -> str | None:
    for col in entity.columns:
        if type_kind(col.type) in ("timestamp", "date") and re.search(
            r"signup|sign_up|created|registered|joined|opened", col.name, re.IGNORECASE
        ):
            return col.name
    return entity.time_column


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _months_back(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) - months
    return date(total // 12, total % 12 + 1, 1)


def _schedule(horizon: str, as_of: datetime) -> tuple[date, date, date, date]:
    """(first cutoff, last cutoff, val_from, test_from): monthly cutoffs, the last one with a
    complete label window, three test months and enough validation months for the horizon."""
    dur = parse_duration(horizon)
    last = _month_start(as_of.date())
    while add_duration(last, dur) > as_of.date():
        last = _months_back(last, 1)
    months = math.ceil((add_duration(date(2000, 1, 1), dur) - date(2000, 1, 1)).days / 30)
    test_from = _months_back(last, 3)
    val_from = _months_back(test_from, months + 2)
    first = _months_back(last, MONTHS_OF_CUTOFFS + months)
    return first, last, val_from, test_from


def rule_draft(question: str, graph: SchemaGraph, as_of: datetime) -> SpecDraft:
    """The offline drafter: a few question shapes against any schema with the right tables.

    It asks a clarifying question instead of guessing when the question has no recognisable event,
    no time horizon, no table to read the event from, or nothing to measure.
    """
    question = question.strip()
    outcome = next((o for o in _OUTCOMES if o.pattern.search(question)), None)
    if outcome is None:
        return _clarify(question, "I could not tell what should be predicted.")
    horizon = _horizon(question)
    if horizon is None:
        return _clarify(question, "Over what time horizon (for example the next 30 days)?")
    entity, entity_note = _entity(question, graph)
    if entity is None:
        return _clarify(question, "I could not tell who the prediction is about.")
    found = _event_table(graph, entity, outcome.tables)
    if found is None:
        return _clarify(
            question,
            f"I found no table of {'/'.join(outcome.tables)} rows with a time column and a link to "
            f"{entity.name}.",
        )
    table, _edge = found
    created = _created_at(entity)
    first, last, val_from, test_from = _schedule(horizon, as_of)
    assert table.time_column is not None
    eligibility: list[Any] = [f"{created} < :cutoff"] if created else []
    assumptions = [
        f"The horizon is {_days(horizon)}.",
        (
            f"Cutoffs are monthly from {first} to {last}; the last one is the latest whose "
            "label window is complete."
        ),
    ]
    if entity_note:
        assumptions.append(entity_note)
    if outcome.kind == "amount":
        column = next(
            (
                c.name
                for key in _AMOUNT_COLUMNS
                for c in table.columns
                if c.name.lower() == key and type_kind(c.type) in ("integer", "float")
            ),
            None,
        )
        if column is None:
            return _clarify(question, f"I found no amount column in {table.name} to add up.")
        target = {
            "type": "regression",
            "expression": {"table": table.name, "agg": "sum", "column": column},
        }
        metric = None
        assumptions.append(
            f"Spend is the sum of {table.name}.{column}; cancelled rows are not excluded."
        )
        name = f"spend_{horizon}"
    else:
        target = {
            "type": "binary",
            "expression": {"table": table.name, "agg": "count", "compare": outcome.compare},
        }
        metric = "pr_auc"
        name = f"{outcome.name.format(table=table.name)}_{horizon}"
        if outcome.kind == "none":
            assumptions.append(
                f"Churn means no row in {table.name} in the {_days(horizon)} after "
                "the cutoff; every row counts, including cancelled ones."
            )
        else:
            assumptions.append(
                f"The event is any row in {table.name} in the window after the cutoff."
            )
        if outcome.needs_history:
            lookback = table.time_column
            eligibility.append(
                {
                    "exists": {
                        "table": table.name,
                        "where": f"{lookback} >= :cutoff - interval '90 days' AND {lookback} < :cutoff"
                        if outcome.kind == "none"
                        else f"{lookback} < :cutoff",
                    }
                }
            )
            assumptions.append(
                "Only "
                + (
                    f"{_singular(entity.name)}s with a row in {table.name} in the 90 days before the cutoff are scored."
                    if outcome.kind == "none"
                    else f"{_singular(entity.name)}s that already have a row in {table.name} are scored."
                )
            )
    data: dict[str, Any] = {
        "name": re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:60],
        "entity": {
            "table": entity.name,
            "key": entity.primary_key[0],
            **({"created_at": created} if created else {}),
        },
        "eligibility": eligibility,
        "target": target,
        "horizon": horizon,
        "cutoffs": {"start": first.isoformat(), "end": last.isoformat(), "every": "1 month"},
        "split": {"val_from": val_from.isoformat(), "test_from": test_from.isoformat()},
    }
    if metric:
        data["metric"] = metric
    try:
        spec = from_dict(data)
    except SpecError as e:  # a bug in the rules: say so
        return SpecDraft("invalid", question, "fallback", None, assumptions, None, list(e.issues))
    issues = validate_against(spec, graph, as_of)
    status: Status = "invalid" if any(i.severity == "error" for i in issues) else "spec"
    return SpecDraft(status, question, "fallback", spec, assumptions, None, issues)


__all__ = [
    "SpecDraft",
    "answer_schema",
    "describe_spec",
    "draft_spec",
    "rule_draft",
]
