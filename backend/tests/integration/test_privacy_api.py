"""Privacy settings and the prompt log through the API (#48), with the offline stub provider."""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from ai.gateway import AIGateway, PromptRecord
from app.api.deps import get_gateway
from app.core import datasets
from app.main import app
from tests.fixtures.api import API, create_project, load_sample
from tests.fixtures.gateway import recording_gateway

SAMPLE = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"


@pytest.fixture
def gateway() -> Iterator[AIGateway]:
    gw = recording_gateway()
    app.dependency_overrides[get_gateway] = lambda: gw
    yield gw
    app.dependency_overrides.pop(get_gateway, None)


async def suggest(client, project_id: str):
    version = (await load_sample(client, project_id))["data_version_id"]
    return await client.get(
        f"{API}/projects/{project_id}/datasets/{version}/suggestions",
        params={"target_column": "Churn", "objective": "Predict churn"},
    )


async def test_privacy_defaults_and_round_trip(client, project_id):
    url = f"{API}/projects/{project_id}/privacy"
    body = (await client.get(url)).json()
    assert body["level"] == "schema_and_stats" and body["never_send"] == []
    assert body["warning"] is None and set(body["level_summaries"]) == {
        "schema_only",
        "schema_and_stats",
        "allow_category_labels",
        "allow_sample_values",
    }

    put = await client.put(
        url, json={"level": "schema_only", "never_send": [" Email ", "orders.total", "email", ""]}
    )
    assert put.status_code == 200, put.text
    assert put.json()["never_send"] == ["Email", "orders.total"]  # trimmed, de-duplicated, sorted
    assert (await client.get(url)).json() == put.json()

    warned = await client.put(url, json={"level": "allow_sample_values", "never_send": []})
    assert "cell values" in warned.json()["warning"]


async def test_privacy_rejects_bad_input_and_unknown_projects(client, project_id):
    url = f"{API}/projects/{project_id}/privacy"
    assert (await client.put(url, json={"level": "everything"})).status_code == 422
    assert (
        await client.put(url, json={"level": "schema_only", "never_send": ["a.b.c"]})
    ).status_code == 422
    assert (await client.get(f"{API}/projects/nope/privacy")).status_code == 404


async def test_settings_are_per_project(client, project_id):
    other = await create_project(client, "Other")
    await client.put(
        f"{API}/projects/{project_id}/privacy", json={"level": "schema_only", "never_send": ["x"]}
    )
    assert (await client.get(f"{API}/projects/{other}/privacy")).json()[
        "level"
    ] == "schema_and_stats"


async def test_default_prompts_hold_no_value_of_the_uploaded_csv(client, project_id, gateway):
    """The planner's prompt carries schema and aggregates; none of the file's text values."""
    records: list[PromptRecord] = []
    gateway.add_prompt_hook(records.append)
    resp = await suggest(client, project_id)
    assert resp.status_code == 200, resp.text
    assert records, "the planner sent no prompt"

    frame = pd.read_csv(SAMPLE)
    values = {
        str(v)
        for col in frame.select_dtypes(exclude="number")
        for v in frame[col].dropna().unique()
    }
    # Numbers stored as text (TotalCharges) can equal a statistic of another column.
    values = {v for v in values if len(v) >= 5 and not v.replace(".", "").isdigit()}
    assert {"Month-to-month", "Fiber optic", "Electronic check"} <= values
    for r in records:
        assert [v for v in values if v in r.prompt + r.system] == []
        assert "Table dataset" in r.prompt and "Contract:" in r.prompt  # the schema is there


async def test_never_send_column_is_in_no_prompt_and_the_log_shows_it(client, project_id, gateway):
    await client.put(
        f"{API}/projects/{project_id}/privacy",
        json={"level": "allow_category_labels", "never_send": ["Contract", "customerID"]},
    )
    records: list[PromptRecord] = []
    gateway.add_prompt_hook(records.append)
    assert (await suggest(client, project_id)).status_code == 200

    for r in records:
        text = (r.system + r.prompt).lower()
        assert "contract" not in text and "customerid" not in text and "month-to-month" not in text
        assert "Fiber optic" in r.prompt  # labels of other columns are allowed at this level

    calls = (await client.get(f"{API}/projects/{project_id}/llm-calls")).json()
    detail = (await client.get(f"{API}/projects/{project_id}/llm-calls/{calls[0]['id']}")).json()
    assert (
        detail["manifest"]["excluded_columns"] == 2
        and detail["manifest"]["level"] == "allow_category_labels"
    )
    assert "contract" not in json.dumps(detail).lower()


async def test_every_prompt_is_logged_with_its_full_text(client, project_id, gateway):
    assert (await suggest(client, project_id)).status_code == 200

    listing = await client.get(f"{API}/projects/{project_id}/llm-calls")
    calls = listing.json()
    assert len(calls) == len(gateway.calls) >= 1
    call = calls[0]
    assert call["purpose"] == "planner.hypothesis" and call["provider"] == "stub"
    assert call["decision_mode"] == "fallback" and call["privacy_level"] == "schema_and_stats"
    assert call["prompt_chars"] > 500 and call["tokens_in"] > 0 and call["error"] is None
    assert "prompt" not in call  # the listing has sizes, not the text

    detail = (await client.get(f"{API}/projects/{project_id}/llm-calls/{call['id']}")).json()
    assert detail["prompt"].startswith("Dataset profile:\nTable dataset") and detail["system"]
    assert isinstance(detail["response"], dict) and "name" in detail["response"]
    assert detail["manifest"]["tables"]["dataset"][0]
    assert len(detail["system"]) + len(detail["prompt"]) == call["prompt_chars"]

    # The text is on disk under the project, readable by the owner only.
    path = datasets.PROJECTS_DIR / project_id / "prompts" / f"{call['id']}.json"
    assert path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_a_failed_provider_attempt_is_logged_too(client, project_id, gateway, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(gateway.providers["stub"], "complete_structured", boom)
    resp = await suggest(client, project_id)
    assert resp.status_code == 502

    (call,) = (await client.get(f"{API}/projects/{project_id}/llm-calls")).json()
    assert call["decision_mode"] == "failed" and "provider down" in call["error"]


async def test_log_is_scoped_to_the_project(client, project_id, gateway):
    assert (await suggest(client, project_id)).status_code == 200
    call_id = (await client.get(f"{API}/projects/{project_id}/llm-calls")).json()[0]["id"]
    other = await create_project(client, "Other")
    assert (await client.get(f"{API}/projects/{other}/llm-calls")).json() == []
    assert (await client.get(f"{API}/projects/{other}/llm-calls/{call_id}")).status_code == 404


async def test_chat_prompt_goes_through_the_builder(client, project_id, gateway):
    await client.put(
        f"{API}/projects/{project_id}/privacy",
        json={"level": "schema_only", "never_send": ["tenure"]},
    )
    resp = await client.post(
        f"{API}/projects/{project_id}/chat/ask", json={"query": "Does tenure matter most?"}
    )
    assert resp.status_code == 200, resp.text
    (call,) = (await client.get(f"{API}/projects/{project_id}/llm-calls")).json()
    detail = (await client.get(f"{API}/projects/{project_id}/llm-calls/{call['id']}")).json()
    assert call["purpose"] == "chat.ask"
    assert "tenure" not in detail["prompt"].lower() and "[excluded column]" in detail["prompt"]
