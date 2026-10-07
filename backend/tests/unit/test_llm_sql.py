"""The feature proposer (#56): a scripted model answers, the checks are real.

Each scripted answer in tests/fixtures/stub_feature_proposals.yaml stands for what a model might
say for the demo database's churn task: good features, one that reads the label's own window,
one the baseline already has, one that is almost a copy, a sum over a text column, a feature
that is constant. The proposer's checks (schema, guards, duplicates, execution) decide each one.
"""

# ruff: noqa: F811
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai.context_builder import BuiltPrompt, ContextBuilder
from ai.gateway import LLMResult
from ml.features.baseline import BaselineResult
from ml.features.llm_sql import FeatureProposer, ProposalRecord, sql_hash
from tests.unit.test_dfs import demo, demo_baseline  # noqa: F401  (the demo world and its baseline)

ANSWERS: dict[str, dict[str, Any]] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "fixtures" / "stub_feature_proposals.yaml").read_text()
)


def answer(key: str, **changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(ANSWERS[key])
    out.update(changes)
    return out


def result(data: dict[str, Any], mode: str = "llm") -> LLMResult:
    return LLMResult(
        task_type="sql",
        provider="fake",
        model="fake-1",
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        latency_ms=0.0,
        decision_mode=mode,  # type: ignore[arg-type]
        prompt_id="p",
        data=data,
    )


@dataclass
class Scripted:
    """Answers with the next scripted object and keeps every prompt it was given."""

    answers: list[dict[str, Any]]
    mode: str = "llm"
    prompts: list[BuiltPrompt] = field(default_factory=list)

    async def complete_structured_result(
        self, task_type: Any, prompt: BuiltPrompt, schema: dict[str, Any]
    ) -> LLMResult:
        self.prompts.append(prompt)
        return result(self.answers.pop(0) if self.answers else {}, self.mode)


def proposer(
    demo: dict[str, Any],
    baseline: BaselineResult,
    gateway: Scripted,
    **kw: Any,
) -> FeatureProposer:
    return FeatureProposer(
        spec=demo["spec"],
        graph=demo["graph"],
        tables=demo["tables"],
        baseline=baseline,
        gateway=gateway,  # type: ignore[arg-type]
        builder=ContextBuilder(),
        **kw,
    )


# -- features that should reach execution ------------------------------------------------------


async def test_good_proposals_reach_execution_and_are_remembered(demo, demo_baseline) -> None:
    gateway = Scripted([answer("good_refunds_14d"), answer("good_cancelled_share")])
    p = proposer(demo, demo_baseline, gateway)
    first, second = await p.run(2)
    for record, name in ((first, "refund_count_14d"), (second, "cancelled_order_share_180d")):
        assert record.status == "proposed" and record.reached_execution, record.reasons
        assert record.stage is None and not record.repaired and len(gateway.prompts) == 2
        assert record.values is not None and len(record.values) == len(demo_baseline.labels)
        assert record.proposal is not None and record.proposal.name == name
        assert record.sql and "l.cutoff_time" in record.sql
    assert {"refund_count_14d", "cancelled_order_share_180d"} <= p.names
    # the second prompt knows the first feature is taken
    assert "refund_count_14d" in gateway.prompts[1].text
    print(
        "\nproposed and executed: "
        + ", ".join(
            f"{r.proposal.name} ({r.values.notna().mean():.0%} non-null)" for r in (first, second)
        )  # type: ignore[union-attr]
    )


# -- features that must be stopped, by the check chain -----------------------------------------


async def test_the_leaky_proposal_is_rejected_by_the_point_in_time_guard(
    demo, demo_baseline
) -> None:
    gateway = Scripted([answer("leaky_sql"), answer("leaky_sql", name="orders_soon")])
    p = proposer(demo, demo_baseline, gateway, allow_free_sql=True)
    record = await p.propose()
    assert record.status == "rejected_guard" and record.stage == "guard"
    assert any(r.startswith("reads_future") for r in record.reasons), record.reasons
    assert record.values is None and record.repaired
    assert [a.stage for a in record.attempts] == ["guard", "guard"]  # asked again, still leaky
    print(f"\nleaky proposal: {record.stage} -> {record.reasons[0][:120]}")


async def test_free_sql_is_refused_unless_the_project_enables_it(demo, demo_baseline) -> None:
    gateway = Scripted([answer("leaky_sql"), answer("good_refunds_14d")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert record.attempts[0].stage == "schema"
    assert "free SQL is not enabled" in record.attempts[0].reasons[0]
    assert record.status == "proposed" and record.repaired  # the repair gave a spec


async def test_a_copy_of_a_feature_in_use_is_a_duplicate(demo, demo_baseline) -> None:
    gateway = Scripted([answer("duplicate_ir"), answer("duplicate_ir", name="again")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert record.status == "rejected_duplicate" and record.stage == "duplicate"
    assert "already in use" in record.reasons[0]
    print(f"\nduplicate proposal: {record.stage} -> {record.reasons[0]}")


async def test_the_same_query_in_other_words_is_a_duplicate(demo, demo_baseline) -> None:
    existing = next(
        f.candidate.sql for f in demo_baseline.features if f.candidate.name == "orders__count_30d"
    )
    reworded = existing.replace("\n", " ").replace("SELECT", "select").replace("  ", " ")
    assert reworded != existing and sql_hash(reworded) == sql_hash(existing)
    sql_answer = answer("leaky_sql", name="orders_reworded", sql=reworded)
    gateway = Scripted([sql_answer, sql_answer])
    record = await proposer(demo, demo_baseline, gateway, allow_free_sql=True).propose()
    assert record.status == "rejected_duplicate" and "same query" in record.reasons[0]


async def test_a_feature_that_is_almost_a_copy_is_caught_after_it_is_computed(
    demo, demo_baseline
) -> None:
    gateway = Scripted([answer("near_duplicate_ir"), answer("near_duplicate_ir", name="n31")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert record.status == "rejected_duplicate"
    assert record.stage == "duplicate_after_execution"
    assert "correlation" in record.reasons[0] or "equal" in record.reasons[0]
    print(f"\nnear duplicate: {record.reasons[0]}")


async def test_a_constant_feature_is_rejected_at_execution(demo, demo_baseline) -> None:
    gateway = Scripted([answer("constant_ir"), answer("constant_ir", name="c2")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert (record.status, record.stage) == ("rejected_guard", "execution")
    assert "one value" in record.reasons[0]


# -- the repair round --------------------------------------------------------------------------


async def test_a_rejected_proposal_is_sent_back_once_with_the_reason(demo, demo_baseline) -> None:
    gateway = Scripted([answer("wrong_column_ir"), answer("good_refunds_14d")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert len(gateway.prompts) == 2 and record.repaired and record.status == "proposed"
    assert [a.stage for a in record.attempts] == ["schema", None]
    assert record.attempts[0].reasons[0].startswith("column: 'sum' needs a numeric column")
    second = gateway.prompts[1].text
    assert "Your last proposal was rejected" in second and "needs a numeric column" in second
    assert "Your last proposal was rejected" not in gateway.prompts[0].text
    assert len(record.llm) == 2


async def test_there_is_no_third_attempt(demo, demo_baseline) -> None:
    gateway = Scripted(
        [answer("wrong_column_ir"), answer("wrong_column_ir"), answer("good_refunds_14d")]
    )
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert len(gateway.prompts) == 2 and len(gateway.answers) == 1
    assert record.status == "rejected_guard" and record.stage == "schema"


async def test_an_answer_with_neither_ir_nor_sql_is_invalid_and_repaired(
    demo, demo_baseline
) -> None:
    gateway = Scripted([answer("shapeless"), answer("good_cancelled_share")])
    record = await proposer(demo, demo_baseline, gateway).propose()
    assert record.attempts[0].stage == "parse"
    assert "exactly one of 'ir' and 'sql'" in record.attempts[0].reasons[0]
    assert record.status == "proposed"
    broken = Scripted([{"nonsense": 1}, {"nonsense": 2}])
    final = await proposer(demo, demo_baseline, broken).propose()
    assert final.status == "invalid" and final.stage == "parse" and final.proposal is None


# -- what the model is told ---------------------------------------------------------------------


async def test_rejections_are_shown_to_the_next_proposal_and_the_prompt_is_logged_by_purpose(
    demo,
    demo_baseline,
) -> None:
    gateway = Scripted(
        [
            answer("wrong_column_ir"),
            answer("wrong_column_ir"),
            answer("good_refunds_14d"),
        ]
    )
    p = proposer(demo, demo_baseline, gateway)
    await p.propose()
    await p.propose()
    third = gateway.prompts[2]
    assert third.purpose == "features.propose"
    assert (
        "total_status" in third.text and "schema" in third.text
    )  # name and stage of the rejection
    assert "needs a numeric column" in third.text
    assert p.rejected and p.rejected[-1]["name"] == "total_status"


async def test_the_prompt_has_the_task_the_schema_and_the_features_in_use_but_no_cell_values(
    demo,
    demo_baseline,
) -> None:
    gateway = Scripted([answer("good_refunds_14d")])
    p = proposer(demo, demo_baseline, gateway)
    # put the per-category counts first, so the prompt has to describe them
    p.views = [(g, d, 10.0 if g == "category" else i) for g, d, i in p.views]
    await p.propose()
    text = gateway.prompts[0].text
    assert "orders" in text and "customers" in text  # the schema
    assert "A customer is labelled 1" in text  # the task in words
    strongest = max(demo_baseline.features, key=lambda f: f.importance)
    assert strongest.candidate.description in text  # a feature in use
    # per-category counts are summarised: their values come from the data and are not sent
    assert "values not shown" in text
    category = next(f for f in demo_baseline.features if f.candidate.group == "category")
    value = category.candidate.ir.filter[0].value  # type: ignore[union-attr]
    assert f"'{value}'" not in text and f'"{value}"' not in text
    assert "Free SQL" in text and "proposal_number" in text


# -- no real model -------------------------------------------------------------------------------


async def test_without_a_real_model_nothing_is_proposed(demo, demo_baseline) -> None:
    gateway = Scripted([answer("good_refunds_14d")], mode="fallback")
    records: list[ProposalRecord] = await proposer(demo, demo_baseline, gateway).run(5)
    assert [r.status for r in records] == ["no_llm"]  # stops at the first, does not loop
    assert len(gateway.prompts) == 1 and records[0].llm[0]["decision_mode"] == "fallback"
