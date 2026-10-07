"""The task editor's one-call check (#99): problems, canonical YAML and the label preview, unsaved."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ml.tasks.spec import from_yaml
from tests.fixtures.api import API
from tests.fixtures.demo_db import demo_sqlite
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_task_spec import CHURN, changed


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


async def check(client: httpx.AsyncClient, project_id: str, **body: object) -> dict:
    resp = await client.post(tasks(project_id, "check"), json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()  # type: ignore[no-any-return]


async def test_a_valid_spec_returns_its_yaml_and_the_label_preview(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    out = await check(
        client, project_id, yaml=CHURN, connection_id=connection, data_version_id=version
    )
    assert [i for i in out["issues"] if i["severity"] == "error"] == []
    assert from_yaml(out["yaml"]) == from_yaml(CHURN)  # the YAML tab shows the same spec
    assert out["spec"]["horizon"] == "30d"
    preview = out["preview"]
    assert preview["mode"] == "snapshot" and len(preview["cutoffs"]) == 19
    assert preview["feasibility"]["status"] in ("ok", "warn")
    assert preview["table"] is None, "the editor's preview never writes a label table"


async def test_the_form_can_send_the_spec_as_an_object(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    by_yaml = await check(
        client, project_id, yaml=CHURN, connection_id=connection, data_version_id=version
    )
    by_form = await check(
        client,
        project_id,
        spec=json.loads(from_yaml(CHURN).model_dump_json(exclude_none=True)),
        connection_id=connection,
        data_version_id=version,
    )
    assert by_form["yaml"] == by_yaml["yaml"]
    assert by_form["preview"]["total_rows"] == by_yaml["preview"]["total_rows"]


async def test_changing_the_horizon_changes_the_preview(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    # Without an explicit window the horizon is the window: no second place to keep in step.
    plain = changed(CHURN, lambda d: d["target"].pop("window"))
    base = await check(
        client, project_id, yaml=plain, connection_id=connection, data_version_id=version
    )
    longer = await check(
        client,
        project_id,
        yaml=changed(plain, lambda d: d.update(horizon="60d")),
        connection_id=connection,
        data_version_id=version,
    )
    assert longer["spec"]["horizon"] == "60d"
    base_rate = [c["base_rate"] for c in base["preview"]["cutoffs"]]
    longer_rate = [c["base_rate"] for c in longer["preview"]["cutoffs"]]
    assert base_rate != longer_rate


async def test_problems_come_back_with_their_paths_and_no_preview(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    bad = await check(
        client,
        project_id,
        yaml=changed(CHURN, lambda d: d["entity"].update(table="clients")),
        connection_id=connection,
        data_version_id=version,
    )
    assert bad["preview"] is None and bad["yaml"]
    assert "entity.table" in {i["path"] for i in bad["issues"]}

    broken = await check(client, project_id, yaml="name: [", connection_id=connection)
    assert broken["spec"] is None and broken["yaml"] is None and broken["preview"] is None
    assert "not valid YAML" in broken["issues"][0]["message"]

    partial = await check(
        client, project_id, spec={"name": "x", "horizon": "soon"}, connection_id=connection
    )
    assert partial["preview"] is None and {i["path"] for i in partial["issues"]} >= {
        "entity",
        "target",
    }


async def test_a_check_saves_nothing(
    client: httpx.AsyncClient, project_id: str, connection: str
) -> None:
    version = await take_snapshot(client, project_id, connection, "2025-01-01T00:00:00Z")
    await check(client, project_id, yaml=CHURN, connection_id=connection, data_version_id=version)
    assert (await client.get(tasks(project_id))).json() == []


async def test_a_wrong_connection_is_404(client: httpx.AsyncClient, project_id: str) -> None:
    resp = await client.post(
        tasks(project_id, "check"), json={"yaml": CHURN, "connection_id": "does-not-exist"}
    )
    assert resp.status_code == 404
