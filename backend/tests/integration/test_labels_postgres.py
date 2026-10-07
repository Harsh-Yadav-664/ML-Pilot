"""Labels run read-only on a real Postgres through the SQL guard (#50) and agree with DuckDB."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ml.data.schema_graph import build_schema_graph
from ml.data.sources import ConnectionSpec, open_source
from ml.tasks.labels import run_on_source
from ml.tasks.spec import from_yaml
from tests.fixtures.demo_db import demo_duckdb
from tests.unit.test_labels import CHURN

AS_OF = datetime(2025, 1, 1, tzinfo=UTC)


def test_postgres_labels_match_the_same_data_in_duckdb(demo_pg, tmp_path: Path) -> None:
    _, ro, _ = demo_pg
    pg = open_source(ro)
    spec = from_yaml(CHURN)
    on_pg = run_on_source(spec, build_schema_graph(pg), pg, AS_OF)

    duck = open_source(
        ConnectionSpec(dialect="duckdb", database=str(demo_duckdb(tmp_path / "demo.duckdb")))
    )
    on_duck = run_on_source(spec, build_schema_graph(duck), duck, AS_OF)

    assert on_pg.dialect == "postgres" and on_duck.dialect == "duckdb"
    assert on_pg.counts == on_duck.counts
    assert on_pg.coverage == on_duck.coverage and len(on_pg.coverage) >= 5  # the feasibility check
    assert on_pg.total_rows > 1000
    print(f"\npostgres == duckdb: {len(on_pg.counts)} cutoffs, {on_pg.total_rows} label rows")
