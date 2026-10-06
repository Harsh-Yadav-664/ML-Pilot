"""The Docker image of the demo database (`docker compose -f docker/demo-db/docker-compose.yml up demo-db`, #47).

The CI job `demo-db` starts the compose service and sets MLPILOT_DEMO_PG_*; everywhere else
this skips. In that job MLPILOT_DEMO_PG_REQUIRED=1 makes a missing database a failure.
"""

from __future__ import annotations

import os

import pytest

from ml.data.sources import ConnectionSpec
from tests.fixtures.demo_db import assert_demo_database, demo_module


def _spec(user_var: str, password_var: str) -> ConnectionSpec:
    return ConnectionSpec(
        dialect="postgres",
        host=os.environ["MLPILOT_DEMO_PG_HOST"],
        port=int(os.environ.get("MLPILOT_DEMO_PG_PORT", "5433")),
        database=os.environ.get("MLPILOT_DEMO_PG_DB", "demo"),
        username=os.environ[user_var],
        password=os.environ[password_var],
        ssl_mode="disable",
    )


def test_the_compose_demo_database_has_the_tables_the_roles_and_the_signal() -> None:
    if not os.environ.get("MLPILOT_DEMO_PG_HOST"):
        if os.environ.get("MLPILOT_DEMO_PG_REQUIRED"):
            pytest.fail("MLPILOT_DEMO_PG_REQUIRED is set but the demo database is not configured")
        pytest.skip(
            "no demo database: start docker/demo-db/docker-compose.yml and set MLPILOT_DEMO_PG_*"
        )
    ro = _spec("MLPILOT_DEMO_PG_RO_USER", "MLPILOT_DEMO_PG_RO_PASSWORD")
    rw = _spec("MLPILOT_DEMO_PG_RW_USER", "MLPILOT_DEMO_PG_RW_PASSWORD")
    counts = assert_demo_database(ro, rw, min_customers=10_000)
    print("\n" + ", ".join(f"{t}={n}" for t, n in counts.items()))
    # the container generated the same data as the script does for the same seed
    expected = {t: n for t, (n, _) in demo_module().checksums(demo_module().generate()).items()}
    assert counts == expected
