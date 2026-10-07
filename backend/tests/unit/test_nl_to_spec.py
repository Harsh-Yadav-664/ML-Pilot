"""From a question to a task spec (#53): the rule-based drafter, the repair round, vague questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from ai.context_builder import BuiltPrompt, ContextBuilder
from ai.gateway import LLMResult
from ml.data.schema_graph import SchemaGraph, build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks import nl_to_spec
from ml.tasks.spec import TaskSpec, from_yaml
from tests.fixtures.demo_db import demo_duckdb
from tests.fixtures.gateway import stub_gateway

AS_OF = datetime(2025, 1, 1, tzinfo=UTC)
FIXTURE = Path(__file__).parent.parent / "fixtures" / "nl_questions.yaml"
QUESTIONS: dict[str, Any] = yaml.safe_load(FIXTURE.read_text())


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> SchemaGraph:
    path = demo_duckdb(tmp_path_factory.mktemp("nl") / "demo.duckdb")
    return build_schema_graph(open_source(ConnectionSpec(dialect="duckdb", database=str(path))))


def matches(spec: TaskSpec, expect: dict[str, str]) -> list[str]:
    """What differs between a drafted spec and the expected one (empty: it matches)."""
    ex = spec.target.expression
    got = {
        "entity": spec.entity.table,
        "table": ex.table if ex else None,
        "type": spec.target.type,
        "agg": ex.agg if ex else None,
        "compare": (ex.compare or "").replace(" ", "") if ex else None,
        "column": ex.column if ex else None,
        "horizon": spec.horizon,
    }
    want = {k: (v.replace(" ", "") if k == "compare" else v) for k, v in expect.items()}
    return [f"{k}: expected {v!r}, got {got[k]!r}" for k, v in want.items() if got[k] != v]


@pytest.mark.parametrize("case", QUESTIONS["questions"], ids=lambda c: c["q"][:40])
def test_the_rule_drafter_gives_the_expected_valid_spec(
    case: dict[str, Any], graph: SchemaGraph
) -> None:
    draft = nl_to_spec.rule_draft(case["q"], graph, AS_OF)
    assert draft.status == "spec", (draft.clarifying_question, draft.issues)
    assert draft.decision_mode == "fallback"
    assert draft.spec is not None and draft.errors == []
    assert matches(draft.spec, case["expect"]) == []
    assert from_yaml(draft.yaml or "") == draft.spec  # the YAML it shows is the spec it means
    assert draft.assumptions, "every draft lists its judgement calls"


@pytest.mark.parametrize("question", QUESTIONS["vague"])
def test_a_vague_question_gets_a_question_back_not_a_spec(
    question: str, graph: SchemaGraph
) -> None:
    draft = nl_to_spec.rule_draft(question, graph, AS_OF)
    assert draft.status == "clarify" and draft.spec is None
    assert draft.clarifying_question and draft.clarifying_question.endswith('"')


def test_cutoffs_fit_the_data_and_the_last_label_window_is_complete(graph: SchemaGraph) -> None:
    draft = nl_to_spec.rule_draft(
        "Which customers will stop ordering in the next 90 days?", graph, AS_OF
    )
    spec = draft.spec
    assert spec is not None
    assert [i for i in draft.issues if i.severity == "error"] == []
    last = datetime.fromisoformat(f"{spec.cutoffs.end}T00:00:00+00:00")
    assert nl_to_spec.add_duration(last.date(), nl_to_spec.parse_duration("90d")) <= AS_OF.date()
    assert spec.split.val_from < spec.split.test_from <= spec.cutoffs.end


def test_describe_spec_is_plain_words(graph: SchemaGraph) -> None:
    draft = nl_to_spec.rule_draft(
        "Which customers will stop ordering in the next 30 days?", graph, AS_OF
    )
    assert draft.spec is not None
    text = nl_to_spec.describe_spec(draft.spec)
    assert "labelled 1" in text and "no row in orders" in text and "30 days" in text
    assert "Cutoffs: every 1 month" in text


async def test_with_the_offline_stub_the_draft_is_recorded_as_fallback(graph: SchemaGraph) -> None:
    draft = await nl_to_spec.draft_spec(
        "Which customers will ask for a refund in the next 30 days?",
        graph,
        gateway=stub_gateway(),
        builder=ContextBuilder(),
        as_of=AS_OF,
    )
    assert draft.status == "spec" and draft.decision_mode == "fallback"
    assert draft.llm and draft.llm[0]["decision_mode"] == "fallback"


async def test_an_empty_question_is_refused(graph: SchemaGraph) -> None:
    with pytest.raises(ValueError, match="Ask a question"):
        await nl_to_spec.draft_spec(
            "  ", graph, gateway=stub_gateway(), builder=ContextBuilder(), as_of=AS_OF
        )


# -- an LLM that answers: validation and the one repair round ----------------------------------


def _result(data: dict[str, Any]) -> LLMResult:
    return LLMResult(
        task_type="spec",
        provider="fake",
        model="fake-1",
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        latency_ms=0.0,
        decision_mode="llm",
        prompt_id="p",
        data=data,
    )


@dataclass
class ScriptedGateway:
    """Answers with the next scripted object and keeps the prompts it was given."""

    answers: list[dict[str, Any]]
    prompts: list[BuiltPrompt] = field(default_factory=list)

    async def complete_structured_result(
        self, task_type: Any, prompt: BuiltPrompt, schema: dict[str, Any]
    ) -> LLMResult:
        self.prompts.append(prompt)
        return _result(self.answers.pop(0))


def _good_spec(graph: SchemaGraph) -> dict[str, Any]:
    draft = nl_to_spec.rule_draft(
        "Which customers will ask for a refund in the next 30 days?", graph, AS_OF
    )
    assert draft.spec is not None
    return draft.spec.model_dump(mode="json", exclude_none=True)


async def _draft(gateway: ScriptedGateway, graph: SchemaGraph) -> nl_to_spec.SpecDraft:
    return await nl_to_spec.draft_spec(
        "Who will ask for a refund within a month?",
        graph,
        gateway=gateway,  # type: ignore[arg-type]
        builder=ContextBuilder(),
        as_of=AS_OF,
    )


async def test_a_valid_llm_answer_is_used_as_is(graph: SchemaGraph) -> None:
    gateway = ScriptedGateway([{"spec": _good_spec(graph), "assumptions": ["Any refund counts."]}])
    draft = await _draft(gateway, graph)
    assert (draft.status, draft.decision_mode, draft.repaired) == ("spec", "llm", False)
    assert draft.assumptions == ["Any refund counts."] and len(gateway.prompts) == 1


async def test_errors_are_sent_back_once_and_the_repaired_spec_is_used(graph: SchemaGraph) -> None:
    broken = _good_spec(graph)
    broken["target"]["expression"]["table"] = "refunds_made"  # not a table of the database
    gateway = ScriptedGateway(
        [
            {"spec": broken, "assumptions": ["a"]},
            {"spec": _good_spec(graph), "assumptions": ["b"]},
        ]
    )
    draft = await _draft(gateway, graph)
    assert (draft.status, draft.repaired) == ("spec", True) and draft.assumptions == ["b"]
    assert len(gateway.prompts) == 2 and len(draft.llm) == 2
    second = gateway.prompts[1].text
    assert "refunds_made" in second and "has these errors" in second


async def test_a_second_failure_is_reported_not_hidden(graph: SchemaGraph) -> None:
    broken = _good_spec(graph)
    broken["target"]["expression"]["table"] = "refunds_made"
    gateway = ScriptedGateway(
        [{"spec": broken, "assumptions": []}, {"spec": broken, "assumptions": []}]
    )
    draft = await _draft(gateway, graph)
    assert draft.status == "invalid" and draft.errors and draft.repaired
    assert len(gateway.prompts) == 2  # one repair, no more


async def test_a_spec_that_breaks_the_format_is_repaired_too(graph: SchemaGraph) -> None:
    gateway = ScriptedGateway(
        [
            {"spec": {"name": "x", "horizon": "soon"}, "assumptions": []},
            {"spec": _good_spec(graph), "assumptions": []},
        ]
    )
    draft = await _draft(gateway, graph)
    assert draft.status == "spec" and draft.repaired


async def test_the_llm_may_ask_instead_of_guessing(graph: SchemaGraph) -> None:
    gateway = ScriptedGateway(
        [{"spec": None, "assumptions": [], "clarifying_question": "Over what horizon?"}]
    )
    draft = await _draft(gateway, graph)
    assert (draft.status, draft.clarifying_question) == ("clarify", "Over what horizon?")
    assert draft.spec is None and len(gateway.prompts) == 1


async def test_the_prompt_has_schema_examples_and_no_raw_rows(graph: SchemaGraph) -> None:
    gateway = ScriptedGateway([{"spec": _good_spec(graph), "assumptions": []}])
    await _draft(gateway, graph)
    text = gateway.prompts[0].text
    for needle in (
        "customers",
        "refunds",
        "Worked examples",
        "Task spec JSON Schema",
        "2025-01-01",
    ):
        assert needle in text
    assert gateway.prompts[0].purpose == "tasks.draft_spec"
