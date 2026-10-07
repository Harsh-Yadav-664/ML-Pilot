"""Task specs through the API (#49): drafts, confirming, and versions that never touch old runs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

import app.db.session as db_session
from app.db.models import Run
from tests.fixtures.api import API
from tests.fixtures.demo_db import demo_sqlite
from tests.unit.test_task_spec import CHURN, REFUND, SPEND, changed


@pytest.fixture
async def connection(client: httpx.AsyncClient, project_id: str, tmp_path: Path) -> str:
    path = demo_sqlite(tmp_path / "demo.sqlite")
    resp = await client.post(
        f"{API}/projects/{project_id}/connections/",
        json={"name": "shop", "dialect": "sqlite", "database": str(path)},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def snapshot_version(client: httpx.AsyncClient, project_id: str, connection: str) -> str:
    """A live version of the demo database as of 2025-01-01: where the data ends."""
    resp = await client.post(
        f"{API}/projects/{project_id}/connections/{connection}/snapshots",
        json={"mode": "live", "tables": ["customers"], "as_of": "2025-01-01T00:00:00Z"},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["version"]["id"])


def body(connection: str, text: str, version: str | None = None) -> dict[str, object]:
    return {"yaml": text, "connection_id": connection, "data_version_id": version}


def url(project_id: str, tail: str = "") -> str:
    return f"{API}/projects/{project_id}/tasks/{tail}".rstrip("/") + ("/" if not tail else "")


async def test_the_schema_endpoint_serves_the_json_schema(
    client: httpx.AsyncClient, project_id: str
) -> None:
    schema = (await client.get(url(project_id, "schema"))).json()
    assert "entity" in schema["properties"] and "horizon" in schema["required"]


async def test_validate_returns_the_parsed_spec_or_every_issue_with_its_path(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    ok = (
        await client.post(url(project_id, "validate"), json=body(connection, CHURN, version))
    ).json()
    assert [i for i in ok["issues"] if i["severity"] == "error"] == []
    assert ok["spec"]["name"] == "churn_30d" and len(ok["schema_fingerprint"]) == 64

    bad_text = changed(
        CHURN, lambda d: (d["entity"].update(table="clients"), d.update(horizon="soon"))
    )
    bad = (
        await client.post(url(project_id, "validate"), json=body(connection, bad_text, version))
    ).json()
    assert {i["path"] for i in bad["issues"]} >= {"entity.table", "horizon"}

    broken = (
        await client.post(url(project_id, "validate"), json=body(connection, "name: [", version))
    ).json()
    assert broken["spec"] is None and "not valid YAML" in broken["issues"][0]["message"]


async def test_create_draft_confirm_and_read_back(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    created = await client.post(url(project_id), json=body(connection, CHURN, version))
    assert created.status_code == 201, created.text
    task = created.json()
    assert (task["name"], task["version"], task["status"]) == ("churn_30d", 1, "draft")
    assert task["used_by_runs"] == 0 and task["confirmed_at"] is None

    confirmed = await client.post(
        url(project_id, f"{task['id']}/confirm"), json={"data_version_id": version}
    )
    assert confirmed.status_code == 200, confirmed.text
    c = confirmed.json()
    assert c["status"] == "confirmed" and c["confirmed_by"] and c["confirmed_at"]
    assert len(c["schema_fingerprint"]) == 64
    assert (await client.get(url(project_id, task["id"]))).json()["status"] == "confirmed"
    listed = (await client.get(url(project_id))).json()
    assert [t["name"] for t in listed] == ["churn_30d"]


async def test_a_spec_with_errors_is_saved_as_a_draft_but_cannot_be_confirmed(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    text = changed(CHURN, lambda d: d["cutoffs"].update(end="2025-01-15"))
    created = (await client.post(url(project_id), json=body(connection, text, version))).json()
    assert created["status"] == "draft"
    assert any(i["path"] == "cutoffs.end" for i in created["issues"])
    refused = await client.post(
        url(project_id, f"{created['id']}/confirm"), json={"data_version_id": version}
    )
    assert refused.status_code == 422
    assert any(i["path"] == "cutoffs.end" for i in refused.json()["detail"]["issues"])
    assert (await client.get(url(project_id, created["id"]))).json()["status"] == "draft"


async def test_a_spec_that_breaks_the_format_is_a_422_with_field_paths_and_is_not_saved(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    resp = await client.post(
        url(project_id), json=body(connection, changed(CHURN, lambda d: d.pop("split")))
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"] == [
        {"path": "split", "message": "is required", "severity": "error"}
    ]
    assert (await client.get(url(project_id))).json() == []


async def test_a_draft_is_edited_in_place(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    task = (await client.post(url(project_id), json=body(connection, CHURN, version))).json()
    edited = changed(CHURN, lambda d: d.update(horizon="14d", description="shorter horizon"))
    resp = await client.put(url(project_id, task["id"]), json=body(connection, edited, version))
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == task["id"] and resp.json()["version"] == 1
    assert resp.json()["spec"]["horizon"] == "14d"
    assert len((await client.get(url(project_id))).json()) == 1


async def test_editing_a_confirmed_spec_that_a_run_used_makes_a_new_version_and_the_run_keeps_the_old_one(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    task = (await client.post(url(project_id), json=body(connection, CHURN, version))).json()
    confirmed = (
        await client.post(
            url(project_id, f"{task['id']}/confirm"), json={"data_version_id": version}
        )
    ).json()

    run_id = str(uuid.uuid4())
    async with db_session.AsyncSessionLocal() as db:
        db.add(Run(id=run_id, project_id=project_id, task_spec_id=confirmed["id"], engine="LGBM"))
        await db.commit()
    assert (await client.get(url(project_id, confirmed["id"]))).json()["used_by_runs"] == 1

    edited = changed(CHURN, lambda d: d["cutoffs"].update(every="2 weeks"))
    resp = await client.put(
        url(project_id, confirmed["id"]), json=body(connection, edited, version)
    )
    assert resp.status_code == 200, resp.text
    new = resp.json()
    assert new["id"] != confirmed["id"] and new["version"] == 2 and new["status"] == "draft"
    assert new["name"] == "churn_30d" and new["used_by_runs"] == 0
    assert yaml.safe_load(new["yaml"])["cutoffs"]["every"] == "2 weeks"

    # The confirmed version, and the run that points at it, are exactly as they were.
    old = (await client.get(url(project_id, confirmed["id"]))).json()
    assert old["status"] == "confirmed" and old["version"] == 1 and old["yaml"] == confirmed["yaml"]
    assert old["confirmed_at"] == confirmed["confirmed_at"] and old["used_by_runs"] == 1
    async with db_session.AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
    assert run is not None and run.task_spec_id == confirmed["id"]

    # A third edit (of the draft) stays version 2; confirming the draft keeps the history.
    again = (
        await client.put(url(project_id, new["id"]), json=body(connection, edited, version))
    ).json()
    assert again["id"] == new["id"] and again["version"] == 2
    listed = (await client.get(url(project_id))).json()
    assert [(t["version"], t["status"]) for t in listed] == [(2, "draft"), (1, "confirmed")]
    latest = (await client.get(url(project_id), params={"latest_only": True})).json()
    assert [t["version"] for t in latest] == [2]
    print(
        f"\nrun {run_id[:8]} still points at v1 {confirmed['id'][:8]}; edit saved v2 {new['id'][:8]}"
    )


async def test_a_task_keeps_its_name_across_versions_and_names_are_unique(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    task = (await client.post(url(project_id), json=body(connection, CHURN, version))).json()
    renamed = await client.put(
        url(project_id, task["id"]),
        json=body(connection, changed(CHURN, lambda d: d.update(name="other_name")), version),
    )
    assert renamed.status_code == 422 and renamed.json()["detail"]["issues"][0]["path"] == "name"
    duplicate = await client.post(url(project_id), json=body(connection, CHURN, version))
    assert duplicate.status_code == 409 and "churn_30d" in duplicate.json()["detail"]
    other = await client.post(url(project_id), json=body(connection, REFUND, version))
    assert other.status_code == 201
    assert {t["name"] for t in (await client.get(url(project_id))).json()} == {
        "churn_30d",
        "refund_30d",
    }


async def test_warnings_do_not_block_confirming(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    task = (await client.post(url(project_id), json=body(connection, SPEND, version))).json()
    assert [i["severity"] for i in task["issues"]] == ["warning"]
    confirmed = await client.post(
        url(project_id, f"{task['id']}/confirm"), json={"data_version_id": version}
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["issues"][0]["path"] == "target.type"


async def test_tasks_belong_to_their_project_and_connection(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await snapshot_version(client, project_id, connection)
    task = (await client.post(url(project_id), json=body(connection, CHURN, version))).json()
    other = (
        await client.post(
            f"{API}/projects/", json={"name": "other", "task_type": "binary_classification"}
        )
    ).json()["id"]
    assert (await client.get(url(other, task["id"]))).status_code == 404
    assert (
        await client.post(url(other), json=body(connection, CHURN))
    ).status_code == 404  # not its connection
    missing_conn = await client.post(url(project_id), json=body("nope", REFUND))
    assert missing_conn.status_code == 404
    unknown_version = await client.post(url(project_id), json=body(connection, REFUND, "0" * 64))
    assert unknown_version.status_code == 404


async def test_without_a_data_version_the_data_is_taken_to_end_now(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    """Cutoffs in 2024 leave room before today (2026), so no 'labels incomplete' error."""
    assert datetime.now(UTC).year >= 2025
    resp = await client.post(url(project_id, "validate"), json=body(connection, CHURN))
    assert [i for i in resp.json()["issues"] if i["severity"] == "error"] == []
