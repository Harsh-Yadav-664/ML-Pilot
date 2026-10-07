"""Steering a running run (#59): approve or veto features, suggest ideas, change settings."""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import pytest
from sqlalchemy import select

import app.db.session as db_session
from app.db.models import Experiment, Feature, Run
from app.jobs import handlers
from app.services import run_service
from ml.experiments.acceptance import DEFAULT_RULE
from tests.integration.test_baseline_run_api import TABLES, connection, started_run  # noqa: F401
from tests.integration.test_run_loop_api import finished, runs_url, test_calls  # noqa: F401
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer

GOOD = ["good_refunds_14d", "good_cancelled_share", "good_ticket_count_60d"]


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_service, "POLL_SECONDS", 0.1)


async def pending(
    client: httpx.AsyncClient, project_id: str, run_id: str, round_: int
) -> dict[str, Any]:
    """Wait until the run is asking about the proposal of ``round_``."""
    for _ in range(600):
        rows = (await client.get(runs_url(project_id, f"/{run_id}/checkpoints"))).json()
        found = [c for c in rows if c["round"] == round_ and c["state"] == "pending"]
        if found:
            return found[0]  # type: ignore[no-any-return]
        await asyncio.sleep(0.1)
    raise AssertionError(f"no question for round {round_}")


async def start(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
    gateway: Any,
    **request: Any,
) -> tuple[str, str]:
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z", TABLES)
    _, run_id = await started_run(client, project_id, connection, version)
    started = await client.post(runs_url(project_id, f"/{run_id}/start"), json=request)
    assert started.status_code == 202, started.text
    return run_id, str(started.json()["job_id"])


def numbers_of(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers_of(v)]
    if isinstance(value, list | tuple):
        return [n for v in value for n in numbers_of(v)]
    return []


def strings_of(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings_of(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in strings_of(v)]
    return []


def unexplained_numbers(text: str, payload: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that are not a number of ``payload`` (to the digits shown).

    Words that come from the payload (names, descriptions) are taken out first: a feature called
    ``ticket_count_60d`` is not a claim about 60. The only constant in the templates is the
    95% of the confidence interval."""
    for s in sorted(strings_of(payload), key=len, reverse=True):
        text = text.replace(s, " ")
    text = text.replace("95% CI", " ")
    values = numbers_of(payload)
    out = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        digits = len(token.split(".")[1]) if "." in token else 0
        tol = 0.5 * 10**-digits + 1e-9
        if not any(abs(float(token) - v) <= tol for v in values):
            out.append(token)
    return out


async def test_a_run_waits_at_each_feature_and_a_veto_keeps_it_out_of_the_model(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    test_calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)  # anything that is tested is accepted
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in GOOD], cost=0.01)
    run_id, job_id = await start(
        client,
        project_id,
        connection,
        monkeypatch,
        gateway,
        max_rounds=3,
        patience=3,
        approval_mode="approve_each_feature",
        checkpoint_timeout_seconds=6,
    )

    # round 1: the run stops and asks, and nothing of the proposal is recorded yet
    first = await pending(client, project_id, run_id, 1)
    assert first["payload"]["name"] == "refund_count_14d" and first["payload"]["sql"]
    assert first["recommended"] == "approve" and first["timeout_seconds"] == 6
    state = (await client.get(runs_url(project_id, f"/{run_id}"))).json()
    assert state["status"] == "running" and state["rounds"] == 0  # paused, not recorded
    done = await client.post(
        runs_url(project_id, f"/{run_id}/checkpoints/{first['id']}"),
        json={"decision": "approve"},
    )
    assert done.status_code == 200 and done.json()["state"] == "approved"
    again = await client.post(
        runs_url(project_id, f"/{run_id}/checkpoints/{first['id']}"), json={"decision": "veto"}
    )
    assert again.status_code == 409  # answered once

    # round 2: veto
    second = await pending(client, project_id, run_id, 2)
    assert second["payload"]["name"] == "cancelled_order_share_180d"
    vetoed = await client.post(
        runs_url(project_id, f"/{run_id}/checkpoints/{second['id']}"),
        json={"decision": "veto", "note": "not interesting"},
    )
    assert vetoed.json()["state"] == "vetoed" and vetoed.json()["note"] == "not interesting"

    # round 3: nobody answers; after the timeout the recommended action is taken
    third = await pending(client, project_id, run_id, 3)
    assert third["payload"]["name"] == "ticket_count_60d"
    job = await finished(client, project_id, job_id)
    assert job["status"] == "succeeded", job

    checkpoints = (await client.get(runs_url(project_id, f"/{run_id}/checkpoints"))).json()
    assert [c["state"] for c in checkpoints] == ["approved", "vetoed", "timeout"]
    assert checkpoints[2]["decided_at"] is not None

    feats = (await client.get(runs_url(project_id, f"/{run_id}/features"))).json()["features"]
    by_name = {f["name"]: f for f in feats if f["kind"] == "llm_sql"}
    assert by_name["refund_count_14d"]["status"] == "accepted"
    assert by_name["cancelled_order_share_180d"]["status"] == "vetoed"
    assert by_name["cancelled_order_share_180d"]["gain"] is None  # never tested
    assert by_name["ticket_count_60d"]["status"] == "accepted"  # the timeout approved it

    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        final = await db.get(Experiment, run.champion_experiment_id)
        assert final is not None
        assert "refund_count_14d" in final.feature_set
        assert "ticket_count_60d" in final.feature_set
        assert "cancelled_order_share_180d" not in final.feature_set
    assert len(test_calls) == 1

    # the narration says what happened, and every number in it is one of the events'
    narration = (await client.get(runs_url(project_id, f"/{run_id}/narration"))).json()["items"]
    texts = [i["text"] for i in narration]
    assert any("you vetoed it" in t for t in texts)
    assert any("no answer in time" in t for t in texts)
    assert any(t.startswith("Waiting for you (round 1)") for t in texts)
    for item in narration:
        assert unexplained_numbers(item["text"], item["payload"]) == [], item
    print(
        "\ncheckpoints: "
        + ", ".join(f"round {c['round']} {c['state']}" for c in checkpoints)
        + f"; features: { {n: f['status'] for n, f in by_name.items()} }"
        + f"; narration lines checked: {len(narration)}"
    )
    for t in texts:
        print("  " + t)


async def test_a_settings_sentence_changes_the_run_only_after_it_is_applied(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    test_calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Costly([answer(k) for k in GOOD], cost=0.01)
    run_id, job_id = await start(
        client,
        project_id,
        connection,
        monkeypatch,
        gateway,
        max_rounds=3,
        patience=3,
        approval_mode="approve_each_feature",
        checkpoint_timeout_seconds=60,
    )
    waiting = await pending(client, project_id, run_id, 1)

    def budget() -> Any:
        return client.get(runs_url(project_id, f"/{run_id}"))

    assert (await budget()).json()["budget"]["max_cost_usd"] is None
    preview = await client.post(
        runs_url(project_id, f"/{run_id}/settings"), json={"message": "budget $1"}
    )
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["applied"] is False and body["unrecognised"] == []
    assert body["changes"] == [{"setting": "max_cost_usd", "from": None, "to": 1.0}]
    assert (await budget()).json()["budget"]["max_cost_usd"] is None  # a preview changes nothing

    applied = await client.post(
        runs_url(project_id, f"/{run_id}/settings/apply"), json={"message": "budget $1"}
    )
    assert applied.json()["applied"] is True
    assert (await budget()).json()["budget"]["max_cost_usd"] == 1.0
    garbled = await client.post(
        runs_url(project_id, f"/{run_id}/settings/apply"), json={"message": "make it nicer"}
    )
    assert garbled.json()["applied"] is False and garbled.json()["unrecognised"]
    await client.post(
        runs_url(project_id, f"/{run_id}/settings/apply"), json={"message": "at most 1 round"}
    )

    await client.post(
        runs_url(project_id, f"/{run_id}/checkpoints/{waiting['id']}"),
        json={"decision": "approve"},
    )
    job = await finished(client, project_id, job_id)
    assert job["status"] == "succeeded", job
    state = (await budget()).json()
    assert state["rounds"] == 1 and state["stop_reason"] == "max_rounds"  # round 2 never began
    assert state["budget"]["max_cost_usd"] == 1.0 and state["budget"]["max_rounds"] == 1
    assert len(gateway.prompts) == 1  # no second proposal was asked for

    narration = (await client.get(runs_url(project_id, f"/{run_id}/narration"))).json()["items"]
    changed = [i for i in narration if i["type"] == "settings_changed"]
    assert changed and changed[0]["payload"]["changes"]["max_cost_usd"] == {"from": None, "to": 1.0}
    assert "max_rounds 3 -> 1" in changed[0]["text"]
    for item in narration:
        assert unexplained_numbers(item["text"], item["payload"]) == [], item
    # a finished run cannot be steered any more
    late = await client.post(
        runs_url(project_id, f"/{run_id}/settings/apply"), json={"message": "budget $5"}
    )
    assert late.status_code == 409
    print(f"\nbudget after the preview: None; after apply: {state['budget']['max_cost_usd']}")
    print(f"narration: {changed[0]['text']}")


async def test_a_suggestion_reaches_the_next_prompt_and_is_checked_like_any_proposal(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Costly([answer("good_refunds_14d"), answer("leaky_sql")], cost=0.01)
    run_id, job_id = await start(
        client,
        project_id,
        connection,
        monkeypatch,
        gateway,
        max_rounds=2,
        patience=2,
        approval_mode="approve_each_feature",
        checkpoint_timeout_seconds=60,
    )
    first = await pending(client, project_id, run_id, 1)
    sent = await client.post(
        runs_url(project_id, f"/{run_id}/suggestions"),
        json={"text": "something about the weekend shoppers"},
    )
    assert sent.status_code == 201 and sent.json()["state"] == "new"
    await client.post(
        runs_url(project_id, f"/{run_id}/checkpoints/{first['id']}"), json={"decision": "veto"}
    )
    job = await finished(client, project_id, job_id)
    assert job["status"] == "succeeded", job

    assert "weekend shoppers" not in gateway.prompts[0].text
    assert "Suggestions from the user" in gateway.prompts[1].text
    assert "something about the weekend shoppers" in gateway.prompts[1].text
    listed = (await client.get(runs_url(project_id, f"/{run_id}/suggestions"))).json()
    assert [(s["state"], s["used_in_round"]) for s in listed] == [("used", 2)]
    # the second answer is free SQL that reads the future: the guard still stops it
    async with db_session.AsyncSessionLocal() as db:
        rows = (await db.scalars(select(Feature).where(Feature.run_id == run_id))).all()
        by_status = {f.name: f.status for f in rows if f.kind == "llm_sql"}
    assert by_status["refund_count_14d"] == "vetoed"
    assert "rejected_guard" in by_status.values()
    narration = (await client.get(runs_url(project_id, f"/{run_id}/narration"))).json()["items"]
    assert any("using your suggestion" in i["text"] for i in narration)
    late = await client.post(
        runs_url(project_id, f"/{run_id}/suggestions"), json={"text": "too late"}
    )
    assert late.status_code == 409
    print(f"\nsuggestion used in round 2; feature statuses: {by_status}")


async def test_steering_needs_a_run_of_this_project(
    client: httpx.AsyncClient, project_id: str
) -> None:
    for tail, method in (
        ("/checkpoints", "get"),
        ("/suggestions", "get"),
        ("/narration", "get"),
    ):
        resp = await getattr(client, method)(
            runs_url(project_id, f"/00000000-0000-0000-0000-000000000000{tail}")
        )
        assert resp.status_code == 404, (tail, resp.text)
