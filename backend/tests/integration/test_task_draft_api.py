"""Question -> draft -> label preview -> confirm, through the API with the offline stub (#53)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.api.deps import get_gateway
from app.main import app
from tests.fixtures.api import API
from tests.fixtures.demo_db import demo_sqlite
from tests.fixtures.gateway import stub_gateway
from tests.integration.test_tasks_api import take_snapshot

QUESTIONS = [
    "Which customers will stop ordering in the next 30 days?",
    "Which customers will ask for a refund in the next 30 days?",
    "How much will each customer spend in the next 60 days?",
]


@pytest.fixture(autouse=True)
def offline_gateway() -> Iterator[None]:
    app.dependency_overrides[get_gateway] = stub_gateway
    yield
    app.dependency_overrides.pop(get_gateway, None)


@pytest.fixture
async def connection(client: httpx.AsyncClient, project_id: str, tmp_path: Path) -> str:
    path = demo_sqlite(tmp_path / "demo.sqlite")
    resp = await client.post(
        f"{API}/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "sqlite", "database": str(path)},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def tasks(project_id: str, tail: str = "") -> str:
    return f"{API}/projects/{project_id}/tasks/{tail}"


async def draft(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    question: str,
    version: str | None = None,
) -> dict:
    resp = await client.post(
        tasks(project_id, "draft"),
        json={"question": question, "connection_id": connection, "data_version_id": version},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("question", QUESTIONS)
async def test_draft_preview_and_confirm_offline(
    client: httpx.AsyncClient, project_id: str, connection: str, question: str
) -> None:
    version = await take_snapshot(
        client,
        project_id,
        connection,
        "2025-01-01T00:00:00Z",
        ["customers", "orders", "refunds", "support_tickets"],
    )
    drafted = await draft(client, project_id, connection, question, version)
    assert drafted["status"] == "spec" and drafted["decision_mode"] == "fallback"
    assert drafted["issues"] == [] or all(i["severity"] == "warning" for i in drafted["issues"])
    assert drafted["description"] and drafted["assumptions"] and drafted["yaml"]
    assert drafted["source"]["question"] == question

    saved = await client.post(
        tasks(project_id),
        json={
            "yaml": drafted["yaml"],
            "connection_id": connection,
            "data_version_id": version,
            "draft_source": drafted["source"],
        },
    )
    assert saved.status_code == 201, saved.text
    task = saved.json()
    assert task["status"] == "draft" and task["draft_source"]["question"] == question
    assert task["draft_source"]["decision_mode"] == "fallback"

    preview = await client.post(
        tasks(project_id, f"{task['id']}/preview-labels"), json={"data_version_id": version}
    )
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["total_rows"] > 0 and body["cutoffs"] and "SELECT" in body["sql"].upper()
    assert body["dropped_cutoffs"] == [], "the drafted cutoffs fit where the data ends"
    assert body["feasibility"]["status"] in ("ok", "warn")

    confirmed = await client.post(
        tasks(project_id, f"{task['id']}/confirm"), json={"data_version_id": version}
    )
    assert confirmed.status_code == 200, confirmed.text
    done = confirmed.json()
    assert done["status"] == "confirmed" and done["confirmed_by"] and done["confirmed_at"]
    assert done["draft_source"]["question"] == question, "the confirmed spec keeps its origin"

    read = (await client.get(tasks(project_id, task["id"]))).json()
    assert read["draft_source"] == task["draft_source"]


async def test_nothing_is_saved_until_the_user_saves(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    await draft(client, project_id, connection, QUESTIONS[0])
    assert (await client.get(tasks(project_id))).json() == []


async def test_an_ambiguous_question_gets_a_clarifying_question(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    drafted = await draft(client, project_id, connection, "predict customers")
    assert drafted["status"] == "clarify"
    assert drafted["spec"] is None and drafted["yaml"] is None
    assert drafted["clarifying_question"]


async def test_a_blank_question_and_a_wrong_connection_are_errors(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    blank = await client.post(
        tasks(project_id, "draft"), json={"question": "", "connection_id": connection}
    )
    assert blank.status_code == 422
    missing = await client.post(
        tasks(project_id, "draft"),
        json={"question": QUESTIONS[0], "connection_id": "does-not-exist"},
    )
    assert missing.status_code == 404


async def test_cutoffs_are_chosen_to_fit_where_the_data_ends(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    """The demo data ends in 2024/2025; with today's date as the end the cutoffs would run into
    a year of empty windows. The draft reads where the data ends from the column statistics."""
    drafted = await draft(client, project_id, connection, QUESTIONS[0])
    assert drafted["data_ends"] < "2025-06-01"
    assert drafted["spec"]["cutoffs"]["end"] < drafted["data_ends"]
