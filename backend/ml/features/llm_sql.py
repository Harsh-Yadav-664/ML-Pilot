"""An LLM proposes one feature at a time; deterministic code decides whether it may run (#56).

``FeatureProposer.propose`` asks for a single feature, as a typed spec (``FeatureIR``, #100) or,
only if the project enables it, as free SQL. The answer is data. It goes through a chain of
checks, in this order, and the first failure ends it:

1. **schema**: the spec validates against the schema graph (every problem has a field path);
   free SQL is refused unless enabled;
2. **guard**: the spec compiles and the SQL passes the SQL guard and the point-in-time guard
   (a free-SQL query that reads rows after the cutoff is rejected by name, not repaired);
3. **duplicate**: the same spec as one in use, or the same normalised SQL;
4. **execution**: the feature is computed on the label rows (no failure, one row per label row,
   not constant on the training rows);
5. **duplicate after execution**: equal to, or correlated above 0.98 with, a feature in use.

A proposal that fails a check is sent back once, with the reason, and the answer to that is
checked the same way. Every attempt is kept, so the evidence report can show what was tried.
``reached execution`` means ``status == "proposed"``: the feature is computed and waiting for the
gain test, which is not decided here (and never by the LLM).

When no real LLM answered (the offline stub, or every provider failed) nothing is proposed:
``status == "no_llm"``. The DFS baseline (#55) is the feature set without a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb
import pandas as pd
import pyarrow as pa
import sqlglot
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlglot.errors import SqlglotError

from ai.context_builder import ContextBuilder, PromptBuilder
from ai.context_inputs import dataset_from_graph
from ai.gateway import AIGateway
from ai.router import TaskType
from ml.data.profiling.db_stats import TableStats
from ml.data.schema_graph import SchemaGraph
from ml.features import prompts
from ml.features.baseline import (
    BaselineResult,
    flatten_graph,
    same_column,
    typed_times,
)
from ml.features.compile import compile
from ml.features.ir import NAME, FeatureIR, FeatureIRError, validate
from ml.tasks.nl_to_spec import describe_spec
from ml.tasks.pit_guard import CUTOFF, ENTITY, check
from ml.tasks.pit_verify import ContractError, run_feature
from ml.tasks.spec import TaskSpec

CORRELATION_DUPLICATE = 0.98
HISTORY_ROWS = 8
CURRENT_FEATURES_SHOWN = 15
MAX_REASON = 400

Status = Literal["proposed", "rejected_guard", "rejected_duplicate", "no_llm", "invalid"]
Stage = Literal["parse", "schema", "guard", "duplicate", "execution", "duplicate_after_execution"]


class Proposal(BaseModel):
    name: str
    rationale: str
    expected_direction: Literal["increase", "decrease", "unknown"] = "unknown"
    ir: FeatureIR | None = None
    sql: str | None = None
    tables_used: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_of_ir_and_sql(self) -> Proposal:
        if (self.ir is None) == (self.sql is None):
            raise ValueError("give exactly one of 'ir' and 'sql'")
        return self


def answer_schema() -> dict[str, Any]:
    return Proposal.model_json_schema()


@dataclass
class Attempt:
    proposal: dict[str, Any] | None  # what the model answered (None if it was not an object)
    stage: Stage | None  # where it failed; None if it passed every check
    reasons: list[str]


@dataclass
class ProposalRecord:
    status: Status
    stage: Stage | None
    reasons: list[str]
    proposal: Proposal | None = None
    sql: str | None = None  # the SQL that was guard-checked and (if proposed) executed
    repaired: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    llm: list[dict[str, Any]] = field(default_factory=list)
    values: pd.Series | None = field(repr=False, default=None)  # one value per label row

    @property
    def reached_execution(self) -> bool:
        return self.status == "proposed"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_REASON else text[: MAX_REASON - 1] + "…"


def sql_hash(sql: str, dialect: str = "duckdb") -> str:
    """A hash of the query with its text normalised (case, spacing, quoting)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        text = tree.sql(dialect=dialect, normalize=True)
    except SqlglotError:
        text = " ".join(sql.lower().split())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ir_key(ir: FeatureIR) -> str:
    return ir.model_dump_json(exclude={"name"})


class FeatureProposer:
    """Proposes features for one task on one data version, remembering what was tried."""

    def __init__(
        self,
        *,
        spec: TaskSpec,
        graph: SchemaGraph,
        tables: dict[str, pa.Table],
        baseline: BaselineResult,
        gateway: AIGateway,
        builder: ContextBuilder,
        stats: dict[str, TableStats] | None = None,
        allow_free_sql: bool = False,
    ) -> None:
        if baseline.frame is None or baseline.labels is None or baseline.train is None:
            raise ValueError("the baseline result must carry its feature frame and labels")
        self.spec = spec
        self.graph, self.entity = flatten_graph(graph, spec.entity.table)
        self.tables = typed_times(tables, self.graph)
        self.gateway = gateway
        self.builder = builder
        self.stats = stats
        self.allow_free_sql = allow_free_sql
        self.labels = baseline.labels
        self.keys = baseline.labels[[ENTITY, CUTOFF]]
        self.index = pd.MultiIndex.from_frame(self.keys)
        self.train = baseline.train
        self.sample = self.train[:: max(1, len(self.train) // 5000)]
        self.columns: dict[str, pd.Series] = {
            str(c): baseline.frame[c] for c in baseline.frame.columns
        }
        self.views = [
            (f.candidate.group, f.candidate.description, f.importance) for f in baseline.features
        ]
        self.hashes = {sql_hash(f.candidate.sql) for f in baseline.features}
        self.irs = {ir_key(f.candidate.ir) for f in baseline.features if f.candidate.ir}
        self.names = set(self.columns)
        self.rejected: list[dict[str, str]] = []
        self.records: list[ProposalRecord] = []

    # -- the prompt ---------------------------------------------------------------------------
    def _current_features(self) -> list[dict[str, Any]]:
        """Strongest first. Per-category counts are shown without the category values."""
        shown: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group, description, importance in sorted(self.views, key=lambda v: -v[2]):
            if group == "category":
                description = "Counts of rows per common value of a text column (values not shown)"
            if description in seen:
                continue
            seen.add(description)
            shown.append({"feature": description, "share_of_gain": round(importance, 3)})
            if len(shown) >= CURRENT_FEATURES_SHOWN:
                break
        return shown

    def _prompt(self, index: int, remaining: int, repair: Attempt | None) -> PromptBuilder:
        b: PromptBuilder = (
            self.builder.prompt("features.propose", prompts.system_prompt())
            .text("Prediction task", describe_spec(self.spec))
            .dataset(dataset_from_graph(self.graph, self.stats))
            .facts("Entity table", {"entity_table": self.entity})
            .facts("Features in use, strongest first", self._current_features())
            .facts("Names already in use", sorted(self.names))
            .facts("Rejected so far", self.rejected[-HISTORY_ROWS:])
            .facts(
                "Free SQL",
                {"enabled": self.allow_free_sql, "dialect": "duckdb"},
            )
            .facts("Budget", {"proposal_number": index, "proposals_left": remaining})
            .facts("Feature spec JSON Schema", answer_schema())
        )
        if repair is not None:
            b = b.text(
                "Your last proposal was rejected",
                json.dumps(
                    {"proposal": repair.proposal, "stage": repair.stage, "reasons": repair.reasons},
                    default=str,
                ),
            )
        return b

    # -- the checks ---------------------------------------------------------------------------
    def _static_checks(self, p: Proposal) -> tuple[Stage | None, list[str], str | None]:
        """(stage, reasons, None) for the first failing check, or (None, [], sql) if all pass."""
        if not NAME.match(p.name):
            return (
                "schema",
                ["name: use lower-case letters, digits and underscores, starting with a letter"],
                None,
            )
        if p.name in self.names:
            return "schema", [f"name: {p.name!r} is already used by another feature"], None
        if p.ir is not None:
            if p.ir.entity_table != self.entity:
                return (
                    "schema",
                    [f"entity_table: must be {self.entity!r}, the entity of the task"],
                    None,
                )
            errors = validate(p.ir, self.graph)
            if errors:
                return "schema", [f"{e.field}: {e.message}" for e in errors], None
            try:
                sql = compile(p.ir, self.graph, "duckdb")
            except FeatureIRError as e:
                return "guard", [f"{x.field}: {x.message}" for x in e.errors], None
            if ir_key(p.ir) in self.irs:
                return "duplicate", ["the same spec is already in use under another name"], None
        else:
            assert p.sql is not None
            if not self.allow_free_sql:
                return "schema", ["free SQL is not enabled for this project; give an 'ir'"], None
            result = check(p.sql, self.graph, "duckdb")
            if not result.ok or result.sql is None:
                return "guard", [f"{r.code}: {r.message}" for r in result.reasons], None
            sql = result.sql
        if sql_hash(sql) in self.hashes or (p.sql is not None and sql_hash(p.sql) in self.hashes):
            return "duplicate", ["the same query is already in use"], None
        return None, [], sql

    def _execute(self, sql: str) -> tuple[pd.Series | None, Stage | None, list[str]]:
        try:
            out = run_feature(sql, self.tables, self.keys, self.graph)
        except ContractError as e:
            return None, "execution", [f"the query breaks the contract: {e}"]
        except (duckdb.Error, ValueError) as e:  # a query the checks let through can still fail
            return None, "execution", [f"the query failed: {_clip(str(e))}"]
        values = out.set_index([ENTITY, CUTOFF])["value"].reindex(self.index).reset_index(drop=True)
        column = values if pd.api.types.is_numeric_dtype(values) else values.astype("float64")
        if column.iloc[self.train].nunique(dropna=False) <= 1:
            return None, "execution", ["the feature has one value on every training row"]
        for name, other in self.columns.items():
            why = same_column(
                column.iloc[self.sample], other.iloc[self.sample], CORRELATION_DUPLICATE
            )
            if why:
                return None, "duplicate_after_execution", [f"{why} with {name}"]
        return column, None, []

    def _evaluate(self, answer: dict[str, Any]) -> tuple[ProposalRecord, Attempt]:
        try:
            proposal = Proposal.model_validate(answer)
        except ValidationError as e:
            reasons = [
                f"{'.'.join(str(x) for x in err['loc']) or 'answer'}: {err['msg']}"
                for err in e.errors()
            ]
            attempt = Attempt(answer, "parse", reasons)
            return ProposalRecord("invalid", "parse", reasons, attempts=[attempt]), attempt
        stage, reasons, sql = self._static_checks(proposal)
        if stage is not None:
            status: Status = "rejected_duplicate" if stage == "duplicate" else "rejected_guard"
            attempt = Attempt(answer, stage, reasons)
            return (
                ProposalRecord(status, stage, reasons, proposal, attempts=[attempt]),
                attempt,
            )
        assert sql is not None
        column, stage, reasons = self._execute(sql)
        if stage is not None:
            status = (
                "rejected_duplicate" if stage == "duplicate_after_execution" else "rejected_guard"
            )
            attempt = Attempt(answer, stage, reasons)
            return (
                ProposalRecord(status, stage, reasons, proposal, sql, attempts=[attempt]),
                attempt,
            )
        attempt = Attempt(answer, None, [])
        return ProposalRecord(
            "proposed", None, [], proposal, sql, attempts=[attempt], values=column
        ), attempt

    # -- the loop -----------------------------------------------------------------------------
    async def propose(self, remaining: int = 1) -> ProposalRecord:
        """One feature: ask, check, and if a check fails ask once more with the reason."""
        number = len(self.records) + 1
        first = await self.gateway.complete_structured_result(
            TaskType.SQL, self._prompt(number, remaining, None).build(), answer_schema()
        )
        if first.decision_mode == "fallback":
            record = ProposalRecord(
                "no_llm", None, ["no LLM answered; the baseline features stand"]
            )
            record.llm = [first.meta()]
            self.records.append(record)
            return record
        record, attempt = self._evaluate(first.structured)
        record.llm = [first.meta()]
        if record.status not in ("proposed", "no_llm"):
            again = await self.gateway.complete_structured_result(
                TaskType.SQL, self._prompt(number, remaining, attempt).build(), answer_schema()
            )
            second, _ = self._evaluate(again.structured)
            second.repaired = True
            second.attempts = [*record.attempts, *second.attempts]
            second.llm = [*record.llm, again.meta()]
            record = second
        self._remember(record)
        self.records.append(record)
        return record

    def _remember(self, record: ProposalRecord) -> None:
        p = record.proposal
        if record.status == "proposed" and p is not None and record.sql is not None:
            self.names.add(p.name)
            self.hashes.add(sql_hash(record.sql))
            if p.ir is not None:
                self.irs.add(ir_key(p.ir))
            if record.values is not None:
                self.columns[p.name] = record.values
        elif p is not None:
            self.rejected.append(
                {
                    "name": p.name,
                    "stage": str(record.stage),
                    "reason": _clip("; ".join(record.reasons)),
                }
            )

    async def run(self, count: int) -> list[ProposalRecord]:
        """Up to ``count`` proposals; stops at once if no real LLM is answering."""
        out: list[ProposalRecord] = []
        for i in range(count):
            record = await self.propose(remaining=count - i)
            out.append(record)
            if record.status == "no_llm":
                break
        return out
