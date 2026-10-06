"""Request checks for the self-hosted API: the local access token (see local_token.py)."""

from __future__ import annotations

import hmac
import ipaddress
from functools import lru_cache

from fastapi import HTTPException, Request

from app.core.local_token import load_or_create_token

UNAUTHORIZED = "Missing or wrong access token. Start MLPilot with `python start.py`, or send `Authorization: Bearer <token from ~/.mlpilot/token>`."


@lru_cache(maxsize=1)
def api_token() -> str:
    return load_or_create_token()


async def require_token(request: Request) -> None:
    """Every API route depends on this (the health check doesn't)."""
    scheme, _, supplied = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        supplied.strip().encode(), api_token().encode()
    ):
        raise HTTPException(
            status_code=401, detail=UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"}
        )


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
