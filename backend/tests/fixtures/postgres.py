"""A real Postgres for the connection tests.

CI starts one as a service container and sets MLPILOT_TEST_PG_* (see .github/workflows/ci.yml).
On a laptop without one the tests skip; in CI a missing Postgres fails them instead, so the
integration test can't be skipped by accident.
"""

from __future__ import annotations

import os

import pytest

from ml.data.sources import ConnectionSpec


def pg_spec(**overrides: object) -> ConnectionSpec:
    host = os.environ.get("MLPILOT_TEST_PG_HOST")
    if not host:
        if os.environ.get("CI"):
            pytest.fail("CI must provide a Postgres service (MLPILOT_TEST_PG_HOST is not set)")
        pytest.skip("no Postgres: set MLPILOT_TEST_PG_HOST, _PORT, _USER, _PASSWORD, _DB")
    values: dict[str, object] = {
        "dialect": "postgres",
        "host": host,
        "port": int(os.environ.get("MLPILOT_TEST_PG_PORT", "5432")),
        "username": os.environ["MLPILOT_TEST_PG_USER"],
        "password": os.environ["MLPILOT_TEST_PG_PASSWORD"],
        "database": os.environ.get("MLPILOT_TEST_PG_DB", "postgres"),
        "ssl_mode": "prefer",
    }
    values.update(overrides)
    return ConnectionSpec(**values)  # type: ignore[arg-type]
