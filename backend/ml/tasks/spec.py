"""The prediction task spec (#49): what to predict, for whom, over which window, from which cutoffs.

A spec is a small YAML document (reference: docs/task_spec.md). ``from_yaml`` turns it into a
typed ``TaskSpec`` and refuses unknown keys, so a typo is an error and not a silently ignored
setting. ``validate_against`` checks it against the schema graph of the connected database and
returns every problem at once, each with the path of the field it is about.

The target and the eligibility rules are a structured language (table, aggregation, condition,
comparison), not free SQL: the label builder (#50) generates the window SQL from it, so the label
at a cutoff only ever looks at rows in ``(cutoff, cutoff + horizon]``. ``target.expression_sql`` is
an escape hatch for what the language cannot say; it is accepted with a warning, and the report
says so.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal

import sqlglot
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlglot.errors import SqlglotError

from ml.data.schema_graph import Edge, SchemaGraph, Table, type_kind
from ml.tasks.conditions import ConditionError, parse_condition

MAX_CUTOFFS = 2000
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_DURATION = re.compile(r"^(\d+)\s*(d|day|days|w|week|weeks|month|months)$", re.IGNORECASE)
_COMPARE = re.compile(r"^(=|==|!=|<>|>=|<=|>|<)\s*-?\d+(\.\d+)?$")
_WINDOW = re.compile(r"^:cutoff(?:\s*\+\s*interval\s*'(\d+)\s*(hour|day|week)s?')?$", re.IGNORECASE)

BINARY_METRICS = ("pr_auc", "roc_auc", "f1", "log_loss")
REGRESSION_METRICS = ("mae", "rmse", "r2")
MULTICLASS_METRICS = ("macro_f1", "accuracy", "log_loss")
DEFAULT_METRIC = {"binary": "pr_auc", "regression": "mae", "multiclass": "macro_f1"}
METRICS = {
    "binary": BINARY_METRICS,
    "regression": REGRESSION_METRICS,
    "multiclass": MULTICLASS_METRICS,
}
AGGREGATES = ("count", "sum", "avg", "min", "max", "count_distinct")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- the spec -------------------------------------------------------------------------------


class Entity(Strict):
    table: str = Field(description="Who is predicted about, for example customers")
    key: str = Field(description="The column that identifies one entity")
    created_at: str | None = Field(
        None, description="When the entity came to exist; it is not scored before this"
    )


class ExistsTest(Strict):
    table: str
    where: str | None = Field(
        None, description="Condition on this table's columns, may use :cutoff"
    )
    via: str | None = Field(
        None, description="The column of this table that holds the entity key, if several do"
    )


class ExistsRule(Strict):
    exists: ExistsTest | None = None
    not_exists: ExistsTest | None = None

    @model_validator(mode="after")
    def _one_of(self) -> ExistsRule:
        if (self.exists is None) == (self.not_exists is None):
            raise ValueError("give exactly one of exists or not_exists")
        return self


Eligibility = Annotated[str | ExistsRule, Field(union_mode="left_to_right")]


class TargetExpression(Strict):
    table: str
    agg: Literal["count", "sum", "avg", "min", "max", "count_distinct"]
    column: str | None = None
    where: str | None = None
    compare: str | None = Field(
        None, description="Binary tasks: the test that makes the label 1, such as '= 0' or '>= 3'"
    )
    via: str | None = None


class Window(Strict):
    start: str = ":cutoff"
    end: str | None = Field(
        None, description="':cutoff + interval 'N days''; default is the horizon"
    )


class Target(Strict):
    type: Literal["binary", "regression", "multiclass"] = "binary"
    window: Window | None = None
    expression: TargetExpression | None = None
    expression_sql: str | None = Field(
        None, description="Escape hatch: a SELECT that returns the label; flagged in the report"
    )


class Cutoffs(Strict):
    start: date
    end: date
    every: str = Field(description="For example '1 month', '2 weeks' or '7d'")


class Split(Strict):
    val_from: date
    test_from: date


class TaskSpec(Strict):
    name: str = Field(description="Lower-case letters, digits and underscores")
    description: str | None = None
    entity: Entity
    eligibility: list[Eligibility] = Field(default_factory=list)
    target: Target
    horizon: str = Field(description="How far ahead the label looks, for example '30d'")
    cutoffs: Cutoffs
    split: Split
    metric: str | None = Field(None, description="Default by task type")

    @property
    def metric_name(self) -> str:
        return self.metric or DEFAULT_METRIC[self.target.type]


# -- issues and errors ----------------------------------------------------------------------


@dataclass(frozen=True)
class SpecIssue:
    path: str
    message: str
    severity: Literal["error", "warning"] = "error"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": self.severity}


class SpecError(ValueError):
    """The text is not a task spec. ``issues`` say which field is wrong and why."""

    def __init__(self, issues: list[SpecIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{i.path or 'spec'}: {i.message}" for i in issues))


def from_yaml(text: str) -> TaskSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        line = getattr(getattr(e, "problem_mark", None), "line", None)
        where = f" (line {line + 1})" if line is not None else ""
        raise SpecError([SpecIssue("", f"not valid YAML{where}")]) from None
    return from_dict(data)


def from_dict(data: Any) -> TaskSpec:
    if not isinstance(data, dict):
        raise SpecError(
            [SpecIssue("", "a task spec is a mapping of fields (name, entity, target, ...)")]
        )
    try:
        return TaskSpec.model_validate(data)
    except ValidationError as e:
        issues = []
        for err in e.errors(include_input=False, include_url=False):
            path = ".".join(str(p) for p in err["loc"] if not isinstance(p, int) or True)
            path = re.sub(r"\.(\d+)", r"[\1]", path)
            issues.append(SpecIssue(path, _plain(err["type"], err["msg"])))
        raise SpecError(issues) from None


def _plain(kind: str, message: str) -> str:
    if kind == "missing":
        return "is required"
    if kind == "extra_forbidden":
        return "is not a known field"
    return message.removeprefix("Value error, ")


def to_yaml(spec: TaskSpec) -> str:
    return yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False)


def json_schema() -> dict[str, Any]:
    """For the spec editor (#99) and the LLM's structured output (#53)."""
    return TaskSpec.model_json_schema()


# -- durations and cutoffs ------------------------------------------------------------------


@dataclass(frozen=True)
class Duration:
    n: int
    unit: Literal["day", "week", "month"]

    def __str__(self) -> str:
        return f"{self.n} {self.unit}{'s' if self.n != 1 else ''}"


def parse_duration(text: str) -> Duration:
    m = _DURATION.match(text.strip())
    if not m:
        raise ValueError(f"{text!r} is not a duration like '30d', '4 weeks' or '1 month'")
    n, unit = int(m.group(1)), m.group(2).lower()
    if n <= 0:
        raise ValueError("a duration must be longer than zero")
    kind: Literal["day", "week", "month"] = (
        "month" if unit.startswith("month") else "week" if unit.startswith("w") else "day"
    )
    return Duration(n, kind)


def add_months(d: date, months: int) -> date:
    index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(d.day, calendar.monthrange(year, month + 1)[1]))


def add_duration(d: date, dur: Duration, times: int = 1) -> date:
    if dur.unit == "month":
        return add_months(d, dur.n * times)
    return d + timedelta(days=dur.n * times * (7 if dur.unit == "week" else 1))


def cutoff_dates(spec: TaskSpec) -> list[date]:
    """start, start + every, ... up to and including end (months keep the start's day of month)."""
    step = parse_duration(spec.cutoffs.every)
    out: list[date] = []
    while len(out) < MAX_CUTOFFS:
        d = add_duration(spec.cutoffs.start, step, len(out))
        if d > spec.cutoffs.end:
            break
        out.append(d)
    return out


# -- validation against the database --------------------------------------------------------


def schema_fingerprint(graph: SchemaGraph) -> str:
    """A hash of what a spec was validated against: tables, columns, types and time columns."""
    import hashlib
    import json

    shape = [
        [t.key, t.time_column, [[c.name, c.type] for c in t.columns]]
        for t in sorted(graph.tables, key=lambda t: t.key)
    ]
    return hashlib.sha256(json.dumps(shape).encode()).hexdigest()


def validate_against(
    spec: TaskSpec, graph: SchemaGraph, as_of: datetime | None = None
) -> list[SpecIssue]:
    """Every problem of ``spec`` for this database, with the path of the field it is about."""
    return _Validator(spec, graph, as_of or datetime.now(UTC)).run()


class _Validator:
    def __init__(self, spec: TaskSpec, graph: SchemaGraph, as_of: datetime) -> None:
        self.spec, self.graph = spec, graph
        self.as_of = as_of.astimezone(UTC) if as_of.tzinfo else as_of.replace(tzinfo=UTC)
        self.issues: list[SpecIssue] = []

    def err(self, path: str, message: str) -> None:
        self.issues.append(SpecIssue(path, message))

    def warn(self, path: str, message: str) -> None:
        self.issues.append(SpecIssue(path, message, "warning"))

    def run(self) -> list[SpecIssue]:
        spec = self.spec
        if not NAME_RE.match(spec.name):
            self.err(
                "name",
                "use 2 to 64 lower-case letters, digits or underscores, starting with a letter",
            )
        entity = self.entity()
        for i, rule in enumerate(spec.eligibility):
            self.eligibility(i, rule, entity)
        horizon = self.duration("horizon", spec.horizon)
        self.target(entity, horizon)
        every = self.duration("cutoffs.every", spec.cutoffs.every)
        self.schedule(horizon, every)
        metrics = METRICS[spec.target.type]
        if spec.metric is not None and spec.metric not in metrics:
            self.err(
                "metric",
                f"{spec.metric!r} does not fit a {spec.target.type} task; use one of {', '.join(metrics)}",
            )
        if spec.target.type != "binary":
            self.warn(
                "target.type",
                f"{spec.target.type} tasks are described but cannot be trained yet; the trainer refuses them",
            )
        return self.issues

    # tables ---------------------------------------------------------------------------------
    def table(self, path: str, name: str) -> Table | None:
        by_key = {t.key: t for t in self.graph.tables}
        if name in by_key:
            return by_key[name]
        same_name = [t for t in self.graph.tables if t.name == name]
        if len(same_name) == 1:
            return same_name[0]
        if len(same_name) > 1:
            self.err(
                path,
                f"table name {name!r} is in several schemas; write one of {sorted(t.key for t in same_name)}",
            )
        else:
            self.err(
                path,
                f"no table {name!r} in the database (tables: {', '.join(sorted(by_key)[:12])})",
            )
        return None

    def column(self, path: str, table: Table, column: str) -> bool:
        if column in {c.name for c in table.columns}:
            return True
        self.err(path, f"table {table.key!r} has no column {column!r}")
        return False

    def entity(self) -> Table | None:
        e = self.spec.entity
        table = self.table("entity.table", e.table)
        if table is None:
            return None
        if self.column("entity.key", table, e.key) and e.key not in table.primary_key:
            self.warn(
                "entity.key",
                f"{e.key!r} is not the primary key of {table.key!r}; each entity must appear once",
            )
        if e.created_at and self.column("entity.created_at", table, e.created_at):
            kind = type_kind(next(c.type for c in table.columns if c.name == e.created_at))
            if kind not in ("timestamp", "date"):
                self.err("entity.created_at", f"{e.created_at!r} is not a date or timestamp column")
        return table

    def condition(
        self, path: str, text: str | None, table: Table | None, *, needs_cutoff: bool = False
    ) -> None:
        if text is None:
            return
        try:
            cond = parse_condition(text, table=table.name if table else None)
        except ConditionError as e:
            self.err(path, str(e))
            return
        if table is not None:
            for col in cond.columns:
                self.column(path, table, col)
        if needs_cutoff and not cond.uses_cutoff:
            self.err(
                path,
                "must compare an event time with :cutoff, otherwise it does not depend on the prediction date",
            )

    def link(self, path: str, child: Table, entity: Table | None, via: str | None) -> Edge | None:
        """The foreign key from ``child`` to the entity table."""
        if entity is None:
            return None
        edges = [
            e for e in self.graph.edges if e.from_table == child.key and e.to_table == entity.key
        ]
        if via is not None:
            if not self.column(f"{path}.via", child, via):
                return None
            edges = [e for e in edges if e.from_columns == [via]]
        if len(edges) == 1:
            return edges[0]
        if not edges:
            self.err(
                path,
                f"{child.key!r} has no foreign key to {entity.key!r}. Only tables with a direct key to "
                "the entity are supported for now; add the relationship on the schema graph if it exists",
            )
        else:
            cols = ", ".join(e.from_columns[0] for e in edges)
            self.err(
                path,
                f"{child.key!r} has several keys to {entity.key!r} ({cols}); name one with via",
            )
        return None

    # eligibility ----------------------------------------------------------------------------
    def eligibility(self, i: int, rule: str | ExistsRule, entity: Table | None) -> None:
        path = f"eligibility[{i}]"
        if isinstance(rule, str):
            self.condition(path, rule, entity)
            return
        kind = "exists" if rule.exists is not None else "not_exists"
        test = rule.exists or rule.not_exists
        assert test is not None
        child = self.table(f"{path}.{kind}.table", test.table)
        if child is None:
            return
        self.link(f"{path}.{kind}", child, entity, test.via)
        self.condition(f"{path}.{kind}.where", test.where, child)
        if child.time_column is None and not _static(child):
            self.err(
                f"{path}.{kind}.table",
                f"{child.key!r} has no event-time column, so rows cannot be limited to before the cutoff",
            )

    # target ---------------------------------------------------------------------------------
    def target(self, entity: Table | None, horizon: Duration | None) -> None:
        t = self.spec.target
        if (t.expression is None) == (t.expression_sql is None):
            self.err("target", "give exactly one of expression or expression_sql")
        if t.window is not None:
            self.window(t.window, horizon)
        if t.expression_sql is not None:
            self.expression_sql(t.expression_sql)
        if t.expression is None:
            return
        ex = t.expression
        child = self.table("target.expression.table", ex.table)
        if child is None:
            return
        self.link("target.expression", child, entity, ex.via)
        if child.time_column is None:
            self.err(
                "target.expression.table",
                f"{child.key!r} has no event-time column, so the label window cannot be applied. Set one on the schema graph",
            )
        elif _static(child):
            self.err(
                "target.expression.table",
                f"{child.key!r} is marked static; the target needs a table of events",
            )
        self.condition("target.expression.where", ex.where, child)
        if ex.agg != "count":
            if ex.column is None:
                self.err("target.expression.column", f"{ex.agg} needs a column")
            elif self.column("target.expression.column", child, ex.column):
                numeric = type_kind(next(c.type for c in child.columns if c.name == ex.column)) in (
                    "integer",
                    "float",
                )
                if ex.agg in ("sum", "avg") and not numeric:
                    self.err(
                        "target.expression.column",
                        f"{ex.agg} needs a numeric column; {ex.column!r} is not",
                    )
        elif ex.column is not None:
            self.column("target.expression.column", child, ex.column)
        if t.type == "binary":
            if ex.compare is None:
                self.err(
                    "target.expression.compare",
                    "a binary target needs the test that makes the label 1, for example '= 0' or '>= 3'",
                )
            elif not _COMPARE.match(ex.compare.strip()):
                self.err(
                    "target.expression.compare",
                    f"{ex.compare!r} is not a comparison like '= 0', '> 2' or '>= 3.5'",
                )
        elif ex.compare is not None:
            self.err(
                "target.expression.compare", f"a {t.type} target is not compared; remove compare"
            )

    def window(self, w: Window, horizon: Duration | None) -> None:
        start = _window_hours(w.start)
        end = _window_hours(w.end) if w.end is not None else None
        if start is None:
            self.err(
                "target.window.start",
                "use ':cutoff' or ':cutoff + interval 'N days'' (hours, days or weeks)",
            )
        if w.end is not None and end is None:
            self.err(
                "target.window.end", "use ':cutoff + interval 'N days'' (hours, days or weeks)"
            )
        if start is None or (w.end is not None and end is None):
            return
        if horizon is None:
            return
        if horizon.unit == "month":
            self.err(
                "target.window",
                "a window cannot be combined with a horizon in months; use days or weeks, or leave the window out",
            )
            return
        limit = horizon.n * (7 if horizon.unit == "week" else 1) * 24
        if end is not None and end > limit:
            self.err(
                "target.window.end",
                f"the window ends {end // 24} days after the cutoff, past the horizon of {horizon}; labels would not be complete by then",
            )
        if end is not None and end <= start:
            self.err("target.window", "the window must end after it starts")

    def expression_sql(self, sql: str) -> None:
        try:
            trees = sqlglot.parse(sql, dialect="postgres")
        except SqlglotError as e:
            self.err("target.expression_sql", f"cannot parse the SQL: {str(e).splitlines()[0]}")
            return
        if len(trees) != 1 or trees[0] is None or trees[0].key != "select":
            self.err("target.expression_sql", "must be a single SELECT")
            return
        if ":cutoff" not in sql:
            self.err(
                "target.expression_sql",
                "must use :cutoff, otherwise the label does not depend on the prediction date",
            )
        self.warn(
            "target.expression_sql",
            "free SQL: it runs through the SQL guard and the point-in-time guard, is not understood by the "
            "label checks, and is flagged in the evidence report",
        )

    # dates ----------------------------------------------------------------------------------
    def duration(self, path: str, text: str) -> Duration | None:
        try:
            return parse_duration(text)
        except ValueError as e:
            self.err(path, str(e))
            return None

    def schedule(self, horizon: Duration | None, every: Duration | None) -> None:
        c, s = self.spec.cutoffs, self.spec.split
        if c.end < c.start:
            self.err("cutoffs.end", f"is before cutoffs.start ({c.start})")
            return
        if horizon is None or every is None:
            return
        dates = cutoff_dates(self.spec)
        if len(dates) >= MAX_CUTOFFS and add_duration(c.start, every, MAX_CUTOFFS) <= c.end:
            self.err("cutoffs.every", f"gives more than {MAX_CUTOFFS} cutoffs; use a longer step")
            return
        last = dates[-1]
        data_end = self.as_of.date()
        if add_duration(last, horizon) > data_end:
            self.err(
                "cutoffs.end",
                f"the last cutoff ({last}) plus the horizon ({horizon}) is after the data ends ({data_end}); "
                "its labels would be incomplete. Move cutoffs.end back to "
                f"{_latest_cutoff(self.spec, horizon, data_end)} or earlier",
            )
        if not (c.start <= s.val_from <= c.end):
            self.err("split.val_from", f"must lie inside the cutoff range {c.start} to {c.end}")
        if not (c.start <= s.test_from <= c.end):
            self.err("split.test_from", f"must lie inside the cutoff range {c.start} to {c.end}")
        if s.test_from <= s.val_from:
            self.err("split.test_from", f"must be after split.val_from ({s.val_from})")
            return
        if add_duration(s.val_from, horizon) > s.test_from:
            self.err(
                "split.test_from",
                f"must be at least one horizon ({horizon}) after split.val_from ({s.val_from}), so that validation labels are known before test starts",
            )
        if c.start <= s.val_from < add_duration(c.start, horizon):
            self.err(
                "split.val_from",
                f"must be at least one horizon ({horizon}) after cutoffs.start ({c.start}), so that training has labels that end before validation",
            )
        train = [d for d in dates if add_duration(d, horizon) <= s.val_from]
        val = [d for d in dates if s.val_from <= d and add_duration(d, horizon) <= s.test_from]
        test = [d for d in dates if d >= s.test_from]
        for name, part in (("train", train), ("validation", val), ("test", test)):
            if not part:
                self.err(
                    "cutoffs",
                    f"no cutoff is left for {name}: cutoffs every {every} with a {horizon} horizon leave none between the split dates",
                )


def _static(table: Table) -> bool:
    return bool(getattr(table, "is_static", False))


def _window_hours(text: str) -> int | None:
    m = _WINDOW.match(text.strip())
    if not m:
        return None
    if m.group(1) is None:
        return 0
    unit = m.group(2).lower()
    return int(m.group(1)) * {"hour": 1, "day": 24, "week": 168}[unit]


def _latest_cutoff(spec: TaskSpec, horizon: Duration, data_end: date) -> date | str:
    best: date | None = None
    for d in cutoff_dates(spec):
        if add_duration(d, horizon) <= data_end:
            best = d
    return best if best is not None else "an earlier date (no cutoff fits)"
