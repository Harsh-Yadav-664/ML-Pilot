"""The demo database in a real Postgres (#47): tables, foreign keys, read-only and write roles.

The data is loaded into a schema of the CI Postgres service. The Docker image itself
(docker compose up demo-db) is checked by test_demo_db_compose.py in its own CI job.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pg8000.native
import pytest

from ml.data.sources import ConnectionSpec
from tests.fixtures.demo_db import assert_demo_database, demo_module, load_demo_postgres
from tests.fixtures.postgres import pg_spec

SCHEMA = "mlpilot_demo_test"
RO, RW = "mlpilot_demo_t_ro", "mlpilot_demo_t_rw"
PASSWORD = "demo-test-pass-3391"


def _admin(spec: ConnectionSpec) -> pg8000.native.Connection:
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
    admin = _admin(admin_spec)
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


def test_the_demo_tables_load_and_the_roles_behave(demo_pg) -> None:
    _, ro, rw = demo_pg
    counts = assert_demo_database(ro, rw, schema=SCHEMA, min_customers=800)
    print("\n" + ", ".join(f"{t}={n}" for t, n in counts.items()))
    assert counts["customers"] == 800 and counts["orders"] > 5000


def test_the_foreign_keys_are_declared_in_postgres_and_absent_in_the_no_fks_copy(demo_pg) -> None:
    admin_spec, _, _ = demo_pg
    admin = _admin(admin_spec)
    query = (
        "SELECT count(*) FROM information_schema.table_constraints "
        "WHERE constraint_type = 'FOREIGN KEY' AND table_schema = :s"
    )
    try:
        with_fks = admin.run(query, s=SCHEMA)[0][0]
        load_demo_postgres(admin, SCHEMA, fks=False)
        without = admin.run(query, s=SCHEMA)[0][0]
        rows = admin.run(f"SELECT count(*) FROM {SCHEMA}.orders")[0][0]
    finally:
        admin.close()
    expected = sum(1 for t in demo_module().SCHEMA for c in t.columns if c.fk)
    assert with_fks == expected == 9
    assert without == 0 and rows > 0
