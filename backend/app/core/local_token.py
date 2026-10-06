"""The local access token every API call must carry (stdlib only, so start.py can use it).

MLPilot is single-user and self-hosted, so this is not user auth: it stops other
websites open in the same browser, and other machines on the network, from using the
API (and the database connections it holds). The token comes from ``MLPILOT_TOKEN``,
else from ``~/.mlpilot/token`` (``MLPILOT_TOKEN_FILE`` overrides the path), which is
created with a random value and owner-only permissions on first use.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

ENV_TOKEN = "MLPILOT_TOKEN"
ENV_TOKEN_FILE = "MLPILOT_TOKEN_FILE"
MIN_LENGTH = 16


def token_file() -> Path:
    return Path(os.environ.get(ENV_TOKEN_FILE) or Path.home() / ".mlpilot" / "token")


def load_or_create_token() -> str:
    """Return the configured token, creating the token file on first use."""
    explicit = os.environ.get(ENV_TOKEN, "").strip()
    if explicit:
        if len(explicit) < MIN_LENGTH:
            raise ValueError(f"{ENV_TOKEN} must be at least {MIN_LENGTH} characters")
        return explicit
    path = token_file()
    if path.exists():
        token = path.read_text().strip()
        if len(token) >= MIN_LENGTH:
            return token
        raise ValueError(f"{path} holds no usable token; delete it to generate a new one")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = secrets.token_urlsafe(32)
    # O_EXCL: if two processes start at once, one creates the file and the other reads it.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return load_or_create_token()
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token
