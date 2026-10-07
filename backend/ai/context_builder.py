"""The one place a prompt about the user's data is built (AGENTS.md rule 6).

Nothing reaches an LLM provider except a `BuiltPrompt`, and only `ContextBuilder` makes those
(`AIGateway` refuses a plain string). The builder takes typed inputs (a dataset description,
facts, free text) and the project's `PrivacyPolicy`, and returns the prompt together with a
`PromptManifest` that says what the prompt contains.

Levels, each adding to the one before:

* `schema_only`: table and column names, types, keys, relationships, event-time columns.
* `schema_and_stats` (default): plus aggregates: row counts, null shares, distinct counts, and
  min, max, mean, standard deviation and percentiles of numeric columns. No category labels.
* `allow_category_labels`: plus the most frequent values of categorical columns.
* `allow_sample_values`: plus a few example cell values, where the caller supplies them.

Columns the user marked `never_send` are left out at every level, and their names are replaced
in free text and facts as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class PrivacyLevel(StrEnum):
    schema_only = "schema_only"
    schema_and_stats = "schema_and_stats"
    allow_category_labels = "allow_category_labels"
    allow_sample_values = "allow_sample_values"


_ORDER = list(PrivacyLevel)

LEVEL_SUMMARY: dict[PrivacyLevel, str] = {
    PrivacyLevel.schema_only: "Names, types, keys and relationships only. No numbers from your data.",
    PrivacyLevel.schema_and_stats: (
        "Schema plus aggregates (counts, null shares, distinct counts, min, max, mean, percentiles)."
        " No category labels, no cell values."
    ),
    PrivacyLevel.allow_category_labels: (
        "Also the most frequent values of categorical columns (for example plan names)."
    ),
    PrivacyLevel.allow_sample_values: "Also a few example cell values.",
}
SAMPLE_VALUES_WARNING = (
    "Example cell values can contain personal or confidential data. They are sent to the LLM"
    " provider you configured. Use a local model (Ollama) if nothing may leave your network."
)

EXCLUDED = "[excluded column]"

# Aggregates that may appear in a prompt, with the names the rest of MLPilot uses. Anything else a
# caller puts in `ColumnContext.stats` is dropped, so a new statistic cannot reach a prompt by accident.
NUMERIC_STATS = ("mean", "stddev", "min", "max", "p1", "p50", "p99", "skew")
COUNT_STATS = ("non_null", "null_fraction", "distinct")
TIME_STATS = ("time_min", "time_max")
ALLOWED_STATS = frozenset(COUNT_STATS + NUMERIC_STATS + TIME_STATS)


@dataclass(frozen=True)
class PrivacyPolicy:
    """A project's privacy choice. `never_send` holds `column` or `table.column` entries."""

    level: PrivacyLevel = PrivacyLevel.schema_and_stats
    never_send: frozenset[str] = frozenset()

    def allows(self, level: PrivacyLevel) -> bool:
        return _ORDER.index(self.level) >= _ORDER.index(level)

    def excludes(self, table: str | None, column: str) -> bool:
        entries = {e.lower() for e in self.never_send}
        return column.lower() in entries or (
            table is not None and f"{table}.{column}".lower() in entries
        )

    def hidden_names(self) -> list[str]:
        """The column parts of every entry, longest first, for replacing them in text."""
        names = {e.rsplit(".", 1)[-1] for e in self.never_send if e.strip()}
        return sorted(names, key=lambda n: (-len(n), n))

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None) -> PrivacyPolicy:
        raw = (settings or {}).get("privacy") or {}
        return cls(
            level=PrivacyLevel(raw.get("level", PrivacyLevel.schema_and_stats)),
            never_send=frozenset(raw.get("never_send", [])),
        )


@dataclass
class ColumnContext:
    name: str
    type: str = ""
    hint: str | None = None
    primary_key: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    category_labels: list[tuple[str, int]] | None = None
    sample_values: list[str] | None = None


@dataclass
class TableContext:
    name: str
    columns: list[ColumnContext]
    row_count: int | None = None
    row_count_estimated: bool = False
    time_column: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    from_table: str
    from_columns: list[str]
    to_table: str
    to_columns: list[str]
    source: str = "declared"


@dataclass
class DatasetContext:
    """What MLPilot knows about a dataset: one or several tables."""

    tables: list[TableContext]
    relationships: list[Relationship] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PromptManifest:
    """What a prompt holds, stored with it. It never names an excluded column."""

    purpose: str
    level: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, list[str]] = field(default_factory=dict)
    stats_used: list[str] = field(default_factory=list)
    excluded_columns: int = 0
    names_replaced: int = 0
    category_labels_included: bool = False
    sample_values_included: bool = False


_ISSUED = object()


@dataclass(frozen=True)
class BuiltPrompt:
    """A prompt produced by `ContextBuilder`. `AIGateway` accepts nothing else."""

    purpose: str
    system: str
    text: str
    manifest: PromptManifest
    project_id: str | None = None
    _issued: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issued is not _ISSUED:
            raise TypeError("BuiltPrompt is created by ContextBuilder, not directly")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
            return f"{value:.4g}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


class PromptBuilder:
    """Collects the parts of one prompt, applying the policy to each."""

    def __init__(self, policy: PrivacyPolicy, purpose: str, system: str, project_id: str | None):
        self.policy = policy
        self.purpose = purpose
        self.system = system
        self.project_id = project_id
        self._parts: list[tuple[str, str]] = []
        self._manifest = PromptManifest(purpose=purpose, level=policy.level.value)
        self._stats_used: set[str] = set()
        self._patterns = [
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", re.IGNORECASE)
            for n in policy.hidden_names()
        ]

    # -- scrubbing ---------------------------------------------------------------------------
    def scrub(self, text: str) -> str:
        """Replace the name of every `never_send` column."""
        for pattern in self._patterns:
            text, n = pattern.subn(EXCLUDED, text)
            self._manifest.names_replaced += n
        return text

    def _scrub_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return {self.scrub(str(k)): self._scrub_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub_value(v) for v in value]
        return value

    # -- parts -------------------------------------------------------------------------------
    def text(self, title: str, body: str) -> PromptBuilder:
        """Free text written by the user or by MLPilot (a question, an objective, a rule summary)."""
        self._add(title, self.scrub(body), "text")
        return self

    def facts(self, title: str, obj: Any) -> PromptBuilder:
        """Structured facts (metrics, history, importances) as JSON."""
        self._add(title, json.dumps(self._scrub_value(obj), indent=2, default=str), "facts")
        return self

    def dataset(self, ctx: DatasetContext, title: str = "Dataset") -> PromptBuilder:
        self._add(title, self._render_dataset(ctx), "dataset")
        return self

    def _add(self, title: str, body: str, kind: str) -> None:
        self._parts.append((title, body))
        self._manifest.sections.append({"kind": kind, "title": title, "chars": len(body)})

    # -- dataset rendering -------------------------------------------------------------------
    def _render_dataset(self, ctx: DatasetContext) -> str:
        policy, manifest = self.policy, self._manifest
        with_stats = policy.allows(PrivacyLevel.schema_and_stats)
        lines: list[str] = []
        kept: dict[str, set[str]] = {}
        for table in ctx.tables:
            cols = [c for c in table.columns if not policy.excludes(table.name, c.name)]
            manifest.excluded_columns += len(table.columns) - len(cols)
            kept[table.name] = {c.name for c in cols}
            manifest.tables[table.name] = [c.name for c in cols]
            head = f"Table {table.name}"
            facts = []
            if with_stats and table.row_count is not None:
                facts.append(
                    f"{'about ' if table.row_count_estimated else ''}{table.row_count:,} rows"
                )
            if table.time_column and not policy.excludes(table.name, table.time_column):
                facts.append(f"event time: {table.time_column}")
            lines.append(head + (f" ({'; '.join(facts)})" if facts else ""))
            lines.extend(f"  note: {self.scrub(n)}" for n in table.notes)
            lines.extend(self._render_column(c, with_stats) for c in cols)
        edges = []
        for rel in ctx.relationships:
            names = [(rel.from_table, c) for c in rel.from_columns] + [
                (rel.to_table, c) for c in rel.to_columns
            ]
            if all(c in kept.get(t, set()) for t, c in names):
                edges.append(
                    f"  {rel.from_table}.{','.join(rel.from_columns)} -> "
                    f"{rel.to_table}.{','.join(rel.to_columns)} ({rel.source})"
                )
        if edges:
            lines.append("Relationships:")
            lines.extend(edges)
        lines.extend(f"Note: {self.scrub(n)}" for n in ctx.notes)
        return "\n".join(lines)

    def _render_column(self, col: ColumnContext, with_stats: bool) -> str:
        policy, manifest = self.policy, self._manifest
        bits = [col.type or "unknown type"]
        if col.hint:
            bits.append(col.hint)
        if col.primary_key:
            bits.append("primary key")
        if with_stats:
            parts = []
            for key in COUNT_STATS + NUMERIC_STATS + TIME_STATS:
                value = col.stats.get(key)
                if key not in ALLOWED_STATS or value is None:
                    continue
                if key == "null_fraction":
                    parts.append(f"nulls {float(value):.1%}")
                else:
                    parts.append(f"{key} {_fmt(value)}")
                self._stats_used.add(key)
            if parts:
                bits.append("; ".join(parts))
        if col.category_labels and policy.allows(PrivacyLevel.allow_category_labels):
            labels = ", ".join(f"{self.scrub(str(v))} ({n:,})" for v, n in col.category_labels)
            bits.append(f"most frequent values: {labels}")
            manifest.category_labels_included = True
        if col.sample_values and policy.allows(PrivacyLevel.allow_sample_values):
            bits.append("examples: " + ", ".join(self.scrub(str(v)) for v in col.sample_values))
            manifest.sample_values_included = True
        return f"  - {col.name}: " + ", ".join(bits)

    # -- result ------------------------------------------------------------------------------
    def build(self) -> BuiltPrompt:
        text = "\n\n".join(f"{title}:\n{body}" if title else body for title, body in self._parts)
        self._manifest.stats_used = sorted(self._stats_used)
        return BuiltPrompt(
            purpose=self.purpose,
            system=self.scrub(self.system),
            text=text,
            manifest=self._manifest,
            project_id=self.project_id,
            _issued=_ISSUED,
        )


class ContextBuilder:
    """Builds prompts for one project under its privacy policy."""

    def __init__(self, policy: PrivacyPolicy | None = None, project_id: str | None = None) -> None:
        self.policy = policy or PrivacyPolicy()
        self.project_id = project_id

    def prompt(self, purpose: str, system: str = "") -> PromptBuilder:
        return PromptBuilder(self.policy, purpose, system, self.project_id)

    def simple(self, purpose: str, text: str, system: str = "") -> BuiltPrompt:
        """A prompt that is only free text."""
        return self.prompt(purpose, system).text("", text).build()


def manifest_dict(manifest: PromptManifest) -> dict[str, Any]:
    return asdict(manifest)
