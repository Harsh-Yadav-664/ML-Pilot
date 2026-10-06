"""The demo database in a real Postgres (#47): tables, foreign keys, read-only and write roles.

The data is loaded into a schema of the CI Postgres service. The Docker image itself
(docker compose up demo-db) is checked by test_demo_db_compose.py in its own CI job.
"""

from __future__ import annotations

from tests.fixtures.demo_db import assert_demo_database, demo_module, load_demo_postgres
from tests.integration.conftest import SCHEMA, admin_connection


def test_the_demo_tables_load_and_the_roles_behave(demo_pg) -> None:
    _, ro, rw = demo_pg
    counts = assert_demo_database(ro, rw, schema=SCHEMA, min_customers=800)
    print("\n" + ", ".join(f"{t}={n}" for t, n in counts.items()))
    assert counts["customers"] == 800 and counts["orders"] > 5000


def test_the_foreign_keys_are_declared_in_postgres_and_absent_in_the_no_fks_copy(demo_pg) -> None:
    admin_spec, _, _ = demo_pg
    admin = admin_connection(admin_spec)
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
