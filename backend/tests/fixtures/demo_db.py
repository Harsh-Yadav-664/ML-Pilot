"""The synthetic e-commerce demo database (docker/demo-db/generate.py, issue #47) for tests.

``demo_data`` generates a small copy in memory; ``demo_sqlite`` and ``demo_duckdb`` write it
to a file (with or without the declared foreign keys); ``load_demo_postgres`` loads it into a
schema of the test Postgres. The generator is a standalone script, loaded here by path.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

GENERATOR = Path(__file__).resolve().parents[3] / "docker" / "demo-db" / "generate.py"
SMALL_CUSTOMERS = 800


@cache
def demo_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mlpilot_demo_generate", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


@cache
def demo_data(customers: int = SMALL_CUSTOMERS, seed: int = 7) -> dict[str, dict[str, Any]]:
    """Treat the arrays as read-only: the result is cached."""
    return demo_module().generate(customers, seed)


def demo_sqlite(path: Path, *, fks: bool = True, customers: int = SMALL_CUSTOMERS) -> Path:
    demo_module().write_sqlite(demo_data(customers), path, fks=fks)
    return path


def demo_duckdb(path: Path, *, fks: bool = True, customers: int = SMALL_CUSTOMERS) -> Path:
    demo_module().write_duckdb(demo_data(customers), path, fks=fks)
    return path


def load_demo_postgres(
    conn: Any, schema: str, *, fks: bool = True, customers: int = SMALL_CUSTOMERS
) -> None:
    """(Re)create the demo tables in ``schema`` of a pg8000 native connection."""
    conn.run(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.run(f"CREATE SCHEMA {schema}")
    conn.run(f"SET search_path TO {schema}")
    try:
        demo_module().load_postgres(conn, demo_data(customers), fks=fks)
    finally:
        conn.run("SET search_path TO DEFAULT")


def assert_demo_database(
    ro: Any, rw: Any, *, schema: str = "public", min_customers: int = 500, check_signal: bool = True
) -> dict[str, int]:
    """What every copy of the demo database in Postgres must satisfy, through MLPilot's sources.

    ``ro`` and ``rw`` are ``ConnectionSpec``s of a read-only and a write-capable role. Returns the
    row count of every table.
    """
    import numpy as np
    import pg8000.exceptions
    import pg8000.native

    from ml.data.sources import open_source

    names = [t.name for t in demo_module().SCHEMA]
    q = f"{schema}."
    reader = open_source(ro)
    tables = {t.name for t in reader.list_tables() if t.schema == schema}
    assert set(names) <= tables, sorted(set(names) - tables)
    counts = {
        name: int(
            reader.query(f"SELECT count(*) AS n FROM {q}{name}", limit=1, timeout_s=30)
            .column("n")
            .to_pylist()[0]
        )
        for name in names
    }
    assert counts["customers"] >= min_customers and all(n > 0 for n in counts.values()), counts

    # The read-only role cannot write (the database refuses it: 42501 insufficient_privilege) ...
    conn = pg8000.native.Connection(
        user=ro.username,
        password=ro.password,
        host=ro.host,
        port=ro.port,
        database=ro.database,
    )
    try:
        with pytest.raises(pg8000.exceptions.DatabaseError) as err:
            conn.run(f"INSERT INTO {q}products VALUES (999999, 'x', 'y', 1)")
        assert err.value.args[0]["C"] == "42501", err.value
    finally:
        conn.close()
    # ... and MLPilot's own check says so, and flags the role that can.
    assert reader.privileges().can_write is False, reader.privileges().notes
    assert open_source(rw).privileges().can_write is True

    if check_signal:
        cutoff, end = "2024-06-01 00:00:00", "2024-07-01 00:00:00"
        table = reader.query(
            "SELECT c.customer_id, MAX(o.ordered_at) AS last_order, "
            f"(SELECT COUNT(*) FROM {q}orders n WHERE n.customer_id = c.customer_id "
            f"AND n.ordered_at > TIMESTAMP '{cutoff}' AND n.ordered_at <= TIMESTAMP '{end}') "
            f"AS next_30d FROM {q}customers c JOIN {q}orders o ON o.customer_id = c.customer_id "
            f"AND o.ordered_at <= TIMESTAMP '{cutoff}' GROUP BY c.customer_id",
            limit=100_000,
            timeout_s=60,
        )
        last = table.column("last_order").to_numpy(zero_copy_only=False).astype("datetime64[s]")
        days = (np.datetime64(cutoff) - last) / np.timedelta64(1, "D")
        label = (np.array(table.column("next_30d").to_pylist()) == 0).astype(int)
        auc = demo_module().rank_auc(days, label)
        print(f"\nrecency AUC {auc:.3f} on {len(days)} customers (Postgres, schema {schema})")
        assert auc > 0.65
    return counts
