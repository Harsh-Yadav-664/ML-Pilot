"""Fixtures shared by the integration tests that need the demo database in a real Postgres."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pg8000.native
import pytest

from ml.data.sources import ConnectionSpec
from tests.fixtures.demo_db import load_demo_postgres
from tests.fixtures.postgres import pg_spec

SCHEMA = "mlpilot_demo_test"
RO, RW = "mlpilot_demo_t_ro", "mlpilot_demo_t_rw"
PASSWORD = "demo-test-pass-3391"


def admin_connection(spec: ConnectionSpec) -> pg8000.native.Connection:
    return pg8000.native.Connection(
        user=spec.username or "",
        password=spec.password,
        host=spec.host or "",
        port=spec.port or 5432,
        database=spec.database,
    )


@pytest.fixture
def demo_pg() -> Iterator[tuple[ConnectionSpec, ConnectionSpec, ConnectionSpec]]:
    """(admin, read-only, read-write) specs, with the demo tables in ``SCHEMA``."""
    admin_spec = pg_spec()
    admin = admin_connection(admin_spec)
    for role in (RO, RW):
        if admin.run("SELECT 1 FROM pg_roles WHERE rolname = :r", r=role):
            admin.run(f"DROP OWNED BY {role}")
            admin.run(f"DROP ROLE {role}")
        admin.run(f"CREATE ROLE {role} LOGIN PASSWORD '{PASSWORD}'")
    load_demo_postgres(admin, SCHEMA)
    for role in (RO, RW):
        admin.run(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {role}")
        admin.run(f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO {role}")
    admin.run(f"GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {SCHEMA} TO {RW}")
    try:
        yield (
            admin_spec,
            dataclasses.replace(admin_spec, username=RO, password=PASSWORD),
            dataclasses.replace(admin_spec, username=RW, password=PASSWORD),
        )
    finally:
        admin.run(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        for role in (RO, RW):
            admin.run(f"DROP OWNED BY {role}")
            admin.run(f"DROP ROLE {role}")
        admin.close()
