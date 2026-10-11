"""A real MySQL for the connection tests.

CI starts one as a service container and sets MLPILOT_TEST_MYSQL_* (see .github/workflows/ci.yml).
Without MLPILOT_TEST_MYSQL_HOST the tests skip, so the shard runners (which have no MySQL) pass;
the backend-checks job runs them with the service and so cannot skip them by accident.
"""

from __future__ import annotations

import os

import pytest

from ml.data.sources import ConnectionSpec


def mysql_spec(**overrides: object) -> ConnectionSpec:
    host = os.environ.get("MLPILOT_TEST_MYSQL_HOST")
    if not host:
        pytest.skip("no MySQL: set MLPILOT_TEST_MYSQL_HOST, _PORT, _USER, _PASSWORD, _DB")
    values: dict[str, object] = {
        "dialect": "mysql",
        "host": host,
        "port": int(os.environ.get("MLPILOT_TEST_MYSQL_PORT", "3306")),
        "username": os.environ["MLPILOT_TEST_MYSQL_USER"],
        "password": os.environ["MLPILOT_TEST_MYSQL_PASSWORD"],
        "database": os.environ.get("MLPILOT_TEST_MYSQL_DB", "mysql"),
        "ssl_mode": "prefer",
    }
    values.update(overrides)
    return ConnectionSpec(**values)  # type: ignore[arg-type]
