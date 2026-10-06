"""#92: every API call needs the local token, and only the configured UI origins get CORS."""

from __future__ import annotations

import os
import stat

import httpx
import pytest

from app.core import local_token
from app.core.security import is_loopback
from app.main import app
from tests.conftest import TEST_TOKEN
from tests.fixtures.api import API


@pytest.fixture
async def anonymous() -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_a_request_without_the_token_gets_401_and_with_it_200(anonymous):
    no_token = await anonymous.get(f"{API}/projects/")
    assert no_token.status_code == 401
    assert no_token.headers["www-authenticate"] == "Bearer"
    wrong = await anonymous.get(f"{API}/projects/", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    ok = await anonymous.get(f"{API}/projects/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert ok.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/projects/",
        "/projects/p/datasets/sample",
        "/projects/p/experiments",
        "/projects/p/agent/auto-optimize",
        "/projects/p/jobs/j",
        "/projects/p/chat/ask",
    ],
)
async def test_every_router_needs_the_token(anonymous, path):
    method = "GET" if path.endswith(("experiments", "/j")) or path == "/projects/" else "POST"
    resp = await anonymous.request(method, f"{API}{path}", json={})
    assert resp.status_code == 401, (path, resp.status_code)


async def test_every_api_route_is_protected():
    """A router added later without the dependency fails here, not in production."""
    from fastapi.routing import APIRoute

    from app.core.security import require_token

    open_routes = {"/health", "/"}
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path in open_routes:
            continue
        calls = {d.call for d in route.dependant.dependencies}
        assert require_token in calls, route.path


async def test_health_needs_no_token(anonymous):
    assert (await anonymous.get("/health")).status_code == 200


PREFLIGHT = {
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "authorization,content-type",
}


async def test_a_cors_preflight_from_another_site_is_refused(anonymous):
    resp = await anonymous.options(
        f"{API}/projects/", headers={"Origin": "http://evil.example", **PREFLIGHT}
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


async def test_a_cors_preflight_from_the_ui_is_allowed(anonymous):
    resp = await anonymous.options(
        f"{API}/projects/", headers={"Origin": "http://localhost:5173", **PREFLIGHT}
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "authorization" in resp.headers["access-control-allow-headers"].lower()


def test_first_start_creates_an_owner_only_token_file(tmp_path, monkeypatch):
    monkeypatch.delenv(local_token.ENV_TOKEN)
    path = tmp_path / "dir" / "token"
    monkeypatch.setenv(local_token.ENV_TOKEN_FILE, str(path))
    token = local_token.load_or_create_token()
    assert len(token) >= local_token.MIN_LENGTH
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert local_token.load_or_create_token() == token  # stable across restarts


def test_a_short_explicit_token_is_refused(monkeypatch):
    monkeypatch.setenv(local_token.ENV_TOKEN, "short")
    with pytest.raises(ValueError, match="at least"):
        local_token.load_or_create_token()


@pytest.mark.parametrize(
    ("host", "loopback"),
    [("127.0.0.1", True), ("localhost", True), ("::1", True), ("0.0.0.0", False)],
)
def test_is_loopback(host, loopback):
    assert is_loopback(host) is loopback
