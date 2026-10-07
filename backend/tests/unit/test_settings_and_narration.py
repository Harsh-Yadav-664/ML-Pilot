"""Reading settings sentences and narrating events (#59): fixed rules, no model."""

from __future__ import annotations

import pytest

from ml.agents.narration import narrate
from ml.agents.settings_nl import parse
from tests.integration.test_run_control_api import unexplained_numbers

CURRENT = {
    "max_cost_usd": None,
    "max_seconds": None,
    "max_proposals": None,
    "max_rounds": 20,
    "patience": 5,
    "approval_mode": "confirm_task",
}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("budget $1", {"max_cost_usd": 1.0}),
        ("set the budget to $0.50", {"max_cost_usd": 0.5}),
        ("stop after 10 minutes", {"max_seconds": 600.0}),
        ("limit it to 2 hours", {"max_seconds": 7200.0}),
        ("at most 5 rounds", {"max_rounds": 5}),
        ("no more than 8 proposals", {"max_proposals": 8}),
        ("patience 3", {"patience": 3}),
        ("stop after 4 rounds without a gain", {"patience": 4}),
        ("ask me before each feature", {"approval_mode": "approve_each_feature"}),
        ("approve every feature", {"approval_mode": "approve_each_feature"}),
        ("don't ask me", {"approval_mode": "auto"}),
        ("budget $2 and at most 6 rounds", {"max_cost_usd": 2.0, "max_rounds": 6}),
    ],
)
def test_sentences_become_typed_changes(message: str, expected: dict[str, object]) -> None:
    diff = parse(message, CURRENT)
    assert {k: v["to"] for k, v in diff.changes.items()} == expected
    assert diff.unrecognised == []
    assert all(v["from"] == CURRENT[k] for k, v in diff.changes.items())


def test_what_is_not_understood_is_reported_and_nothing_is_guessed() -> None:
    for message in ("make it nicer", "budget $0", "at most 0 rounds"):
        diff = parse(message, CURRENT)
        assert diff.empty and diff.unrecognised, message
    mixed = parse("budget $3 and use the red button", CURRENT)
    assert mixed.changes["max_cost_usd"]["to"] == 3.0 and mixed.unrecognised == ["red button"]


def test_a_setting_that_is_already_so_is_no_change() -> None:
    assert parse("at most 20 rounds", CURRENT).empty
    assert parse("budget $1", {**CURRENT, "max_cost_usd": 1.0}).empty
    assert parse("budget $1", CURRENT).summary() != "No setting would change."


EVENTS = [
    ("baseline", {"features": 139, "val_pr_auc": 0.6344, "base_rate": 0.4798}),
    (
        "feature_decision",
        {
            "round": 2,
            "name": "ticket_count_60d",
            "status": "proposed",
            "stage": None,
            "reasons": [],
            "accepted": True,
            "vetoed": False,
            "cost_usd": 0.01,
            "champion_val_pr_auc": 0.6402,
            "gain": {"mean": 0.0123, "ci95": [0.0041, 0.0205], "margin": 0.002, "metric": "pr_auc"},
        },
    ),
    (
        "feature_decision",
        {
            "round": 3,
            "name": "refund_count_14d",
            "status": "proposed",
            "stage": None,
            "reasons": [],
            "accepted": False,
            "vetoed": False,
            "champion_val_pr_auc": 0.6344,
            "gain": {
                "mean": -0.0051,
                "ci95": [-0.01, -0.0002],
                "margin": 0.0079,
                "metric": "pr_auc",
            },
        },
    ),
    (
        "feature_decision",
        {
            "round": 4,
            "name": "orders_soon",
            "status": "rejected_guard",
            "stage": "guard",
            "reasons": ["reads_future: 12 rows"],
            "accepted": False,
            "vetoed": False,
        },
    ),
    (
        "feature_decision",
        {
            "round": 5,
            "name": "x_14d",
            "status": "proposed",
            "stage": None,
            "reasons": [],
            "accepted": False,
            "vetoed": True,
        },
    ),
    ("feature_decision", {"round": 6, "name": None, "status": "no_llm", "reasons": []}),
    (
        "checkpoint",
        {
            "round": 2,
            "summary": {"name": "ticket_count_60d", "description": "Tickets in the last 60 days"},
            "timeout_seconds": 300.0,
            "recommended": "approve",
        },
    ),
    ("checkpoint_answered", {"round": 2, "decision": "veto", "how": "reply"}),
    ("checkpoint_answered", {"round": 3, "decision": "approve", "how": "timeout"}),
    (
        "budget_stop",
        {
            "budget": "cost",
            "limits": {"max_cost_usd": 0.05},
            "used": {"cost_usd": 0.05, "seconds": 41.2, "proposals": 3},
        },
    ),
    (
        "budget_stop",
        {
            "budget": "time",
            "limits": {"max_seconds": 600.0},
            "used": {"cost_usd": 0.0, "seconds": 601.2, "proposals": 3},
        },
    ),
    (
        "budget_stop",
        {
            "budget": "proposals",
            "limits": {"max_proposals": 8},
            "used": {"cost_usd": 0.0, "seconds": 9.0, "proposals": 8},
        },
    ),
    (
        "run_finished",
        {
            "status": "completed",
            "stop_reason": "patience",
            "test": {"pr_auc": 0.4693, "base_rate": 0.338},
            "error": None,
        },
    ),
    (
        "run_finished",
        {"status": "completed", "stop_reason": "patience", "test": None, "error": "not scored"},
    ),
    (
        "settings_changed",
        {
            "changes": {
                "max_cost_usd": {"from": None, "to": 1.0},
                "max_rounds": {"from": 3, "to": 1},
            }
        },
    ),
    ("suggestion", {"round": 2, "text": "orders refunded in the last 90 days"}),
]


@pytest.mark.parametrize(("type_", "payload"), EVENTS)
def test_every_number_in_a_sentence_is_a_number_of_its_event(
    type_: str, payload: dict[str, object]
) -> None:
    text = narrate(type_, payload)  # type: ignore[arg-type]
    assert text, type_
    assert unexplained_numbers(text, payload) == [], text
    print(f"\n{type_}: {text}")


def test_the_check_itself_catches_a_number_that_is_not_in_the_event() -> None:
    payload = {"round": 2, "decision": "veto", "how": "reply"}
    assert unexplained_numbers("Round 2: veto (your answer).", payload) == []
    assert unexplained_numbers("Round 7: veto, gain +0.0123.", payload) == ["7", "+0.0123"]


def test_steps_are_not_narrated() -> None:
    assert narrate("step", {"name": "x", "progress": 0.5}) is None
