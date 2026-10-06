"""Projects are the API's root resource, and the local owner is created once."""

from __future__ import annotations

import asyncio

from tests.fixtures.api import API


async def test_two_requests_can_create_projects_at_the_same_time(client):
    """The default local user is inserted once; a concurrent create must not 500."""
    body = {"name": "Race", "task_type": "binary_classification"}
    first, second = await asyncio.gather(
        client.post(f"{API}/projects/", json=body),
        client.post(f"{API}/projects/", json=body),
    )
    assert (first.status_code, second.status_code) == (201, 201), (first.text, second.text)
    assert first.json()["id"] != second.json()["id"]
    listed = (await client.get(f"{API}/projects/")).json()["items"]
    assert {first.json()["id"], second.json()["id"]} <= {p["id"] for p in listed}
