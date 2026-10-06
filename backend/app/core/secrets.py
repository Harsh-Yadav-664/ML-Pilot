"""Where a saved connection's password lives, and how it is read back.

A connection row holds a ``secret_ref``, never the password:

* ``env:NAME``: the password is in the environment variable ``NAME`` when it is needed.
  Nothing secret is stored by MLPilot.
* ``enc:<token>``: the password encrypted with Fernet (AES-128-CBC + HMAC-SHA256) under the
  key in ``MLPILOT_SECRET_KEY``. Without that key MLPilot refuses to store a password.

Create a key with ``python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"``.
"""

from __future__ import annotations

import os
import re

from cryptography.fernet import Fernet, InvalidToken

from app.core.redaction import register_secret

KEY_VAR = "MLPILOT_SECRET_KEY"
ENV_PREFIX = "env:"
ENC_PREFIX = "enc:"
# The token must fit the connections.secret_ref column (String(255)).
MAX_PASSWORD_BYTES = 100
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
# MLPilot's own secrets: a saved connection must not be able to send them to a database host.
_RESERVED_ENV = re.compile(r"MLPILOT_.*|.*(_API_KEY|SECRET_KEY|_TOKEN)")


class SecretError(Exception):
    """A secret can't be stored or read. The message is safe to show: it never has the secret."""


def _fernet() -> Fernet:
    key = os.environ.get(KEY_VAR)
    if not key:
        raise SecretError(
            f"{KEY_VAR} is not set, so MLPilot will not store a password. Set it to a Fernet key "
            "(see app/core/secrets.py), or use password_env to read the password from an "
            "environment variable instead."
        )
    try:
        return Fernet(key.encode())
    except ValueError:
        raise SecretError(f"{KEY_VAR} is not a valid Fernet key") from None


def validate_env_name(name: str) -> str:
    if not _ENV_NAME.fullmatch(name):
        raise SecretError("password_env must be an upper-case environment variable name")
    if _RESERVED_ENV.fullmatch(name):
        raise SecretError(f"{name} is reserved and can't be used as a database password")
    return name


def env_ref(name: str) -> str:
    return ENV_PREFIX + validate_env_name(name)


def store_secret(password: str) -> str:
    """Encrypt ``password`` and return the ``secret_ref`` to save. Refuses without a key."""
    if not password:
        raise SecretError("The password is empty")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise SecretError(f"The password is longer than {MAX_PASSWORD_BYTES} bytes")
    token = _fernet().encrypt(password.encode()).decode()
    register_secret(password)
    return ENC_PREFIX + token


def resolve_secret(secret_ref: str | None) -> str | None:
    """The password a ``secret_ref`` stands for (None when the connection has none)."""
    if secret_ref is None:
        return None
    if secret_ref.startswith(ENV_PREFIX):
        name = validate_env_name(secret_ref.removeprefix(ENV_PREFIX))
        value = os.environ.get(name)
        if value is None:
            raise SecretError(f"The environment variable {name} is not set")
        register_secret(value)
        return value
    if secret_ref.startswith(ENC_PREFIX):
        try:
            password = _fernet().decrypt(secret_ref.removeprefix(ENC_PREFIX).encode()).decode()
        except InvalidToken:
            raise SecretError(
                f"The saved password can't be decrypted: {KEY_VAR} differs from the key it was "
                "saved with"
            ) from None
        register_secret(password)
        return password
    raise SecretError("Unknown secret reference")


def describe(secret_ref: str | None) -> str | None:
    """What an API response may say about a secret: where it lives, never its value."""
    if secret_ref is None:
        return None
    if secret_ref.startswith(ENV_PREFIX):
        return secret_ref  # the variable name is not secret
    return "stored (encrypted)"
