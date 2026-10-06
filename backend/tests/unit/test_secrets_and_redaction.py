"""Secrets: where a saved password lives and how logs are kept free of it (#43, rule 7)."""

from __future__ import annotations

import logging

import pytest
from cryptography.fernet import Fernet

from app.core import redaction, secrets

PASSWORD = "s3cret-Pass-9f2c"


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    value = Fernet.generate_key().decode()
    monkeypatch.setenv(secrets.KEY_VAR, value)
    return value


def test_a_password_is_refused_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(secrets.KEY_VAR, raising=False)
    with pytest.raises(secrets.SecretError, match="MLPILOT_SECRET_KEY is not set"):
        secrets.store_secret(PASSWORD)


def test_a_stored_password_is_encrypted_and_reads_back(key: str) -> None:
    ref = secrets.store_secret(PASSWORD)
    assert ref.startswith("enc:") and PASSWORD not in ref
    assert len(ref) <= 255  # fits connections.secret_ref
    assert secrets.resolve_secret(ref) == PASSWORD
    assert secrets.describe(ref) == "stored (encrypted)"


def test_the_longest_allowed_password_fits_the_column(key: str) -> None:
    ref = secrets.store_secret("x" * secrets.MAX_PASSWORD_BYTES)
    assert len(ref) <= 255
    with pytest.raises(secrets.SecretError, match="longer than"):
        secrets.store_secret("x" * (secrets.MAX_PASSWORD_BYTES + 1))


def test_a_different_key_cannot_read_the_password_and_the_error_has_no_secret(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = secrets.store_secret(PASSWORD)
    monkeypatch.setenv(secrets.KEY_VAR, Fernet.generate_key().decode())
    with pytest.raises(secrets.SecretError) as err:
        secrets.resolve_secret(ref)
    assert PASSWORD not in str(err.value)


def test_an_env_reference_reads_the_variable_at_use_time(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = secrets.env_ref("PGPASSWORD_SALES")
    assert ref == "env:PGPASSWORD_SALES" and secrets.describe(ref) == ref
    with pytest.raises(secrets.SecretError, match="PGPASSWORD_SALES is not set"):
        secrets.resolve_secret(ref)
    monkeypatch.setenv("PGPASSWORD_SALES", PASSWORD)
    assert secrets.resolve_secret(ref) == PASSWORD


@pytest.mark.parametrize(
    "name", ["MLPILOT_TOKEN", "MLPILOT_SECRET_KEY", "OPENAI_API_KEY", "lowercase", "A B", ""]
)
def test_env_references_cannot_name_mlpilots_own_secrets(name: str) -> None:
    with pytest.raises(secrets.SecretError):
        secrets.env_ref(name)


def test_registered_secrets_and_url_passwords_are_masked(caplog: pytest.LogCaptureFixture) -> None:
    redaction.install()
    redaction.register_secret(PASSWORD)
    caplog.set_level(logging.DEBUG)
    log = logging.getLogger("tests.redaction")
    log.info("connecting with password %s", PASSWORD)
    log.debug("url is postgresql://analyst:other-pw-77@db.example:5432/sales")
    assert PASSWORD not in caplog.text and "other-pw-77" not in caplog.text
    assert "analyst" in caplog.text and "db.example" in caplog.text  # only the secret is masked


def test_a_traceback_that_contains_a_secret_is_masked(caplog: pytest.LogCaptureFixture) -> None:
    redaction.install()
    redaction.register_secret(PASSWORD)
    caplog.set_level(logging.DEBUG)
    try:
        raise RuntimeError(f"driver said: bad password {PASSWORD}")
    except RuntimeError:
        logging.getLogger("tests.redaction").exception("it failed")
    assert PASSWORD not in caplog.text
    assert "RuntimeError" in caplog.text


def test_the_scan_would_notice_a_leak(caplog: pytest.LogCaptureFixture) -> None:
    """Negative control for the log scans: an unregistered secret is NOT masked, so a scan
    for it finds it. This is what makes 'the scan found nothing' mean something."""
    redaction.install()
    caplog.set_level(logging.DEBUG)
    logging.getLogger("tests.redaction").info("value %s", "never-registered-value")
    assert "never-registered-value" in caplog.text


def test_very_short_secrets_are_left_alone(caplog: pytest.LogCaptureFixture) -> None:
    redaction.install()
    redaction.register_secret("ab")
    caplog.set_level(logging.DEBUG)
    logging.getLogger("tests.redaction").info("table abc")
    assert "table abc" in caplog.text
