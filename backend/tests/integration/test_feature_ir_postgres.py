"""Feature IR on a real Postgres (#100): the Postgres SQL gives the same values as the DuckDB SQL.

The 12 hand-written IRs of ``tests/unit/test_feature_ir.py`` are compiled for Postgres (tables in
a non-default schema, so the names are qualified), run against the demo database loaded there,
and compared with the DuckDB result on the same label rows.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ml.data.schema_graph import EdgeRef, SchemaOverrides, apply_overrides, build_schema_graph
from ml.data.sources import open_source
from ml.features.compile import compile
from ml.features.ir import FeatureIR
from ml.tasks.pit_verify import run_feature
from tests.integration.conftest import SCHEMA, admin_connection
from tests.unit.test_feature_ir import CASES, world  # noqa: F401  (the DuckDB side)


def qualified(ir: FeatureIR, prefix: str) -> FeatureIR:
    """The same IR with every table named ``schema.table``, as the schema graph of a schema is."""

    def fix(ref: EdgeRef) -> EdgeRef:
        return EdgeRef(
            from_table=prefix + ref.from_table,
            from_columns=ref.from_columns,
            to_table=prefix + ref.to_table,
            to_columns=ref.to_columns,
        )

    return ir.model_copy(
        update={
            "entity_table": prefix + ir.entity_table,
            "path": [fix(e) for e in ir.path],
            "source_table": prefix + ir.source_table,
            "ratio_to": qualified(ir.ratio_to, prefix) if ir.ratio_to else None,
        }
    )


def same(a: Any, b: Any) -> bool:
    a_null, b_null = pd.isna(a), pd.isna(b)
    if a_null or b_null:
        return bool(a_null and b_null)
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def test_postgres_and_duckdb_agree_on_the_hand_written_irs(demo_pg, world) -> None:  # noqa: F811
    admin_spec, ro, _ = demo_pg
    graph = apply_overrides(
        build_schema_graph(open_source(ro)),
        SchemaOverrides(static_tables={f"{SCHEMA}.order_items": True, f"{SCHEMA}.products": True}),
    )
    labels = world["labels"].sample(n=150, random_state=1)
    admin = admin_connection(admin_spec)
    compared = 0
    try:
        admin.run("CREATE TEMP TABLE __labels (entity_id bigint, cutoff_time timestamp)")
        for entity, cutoff in zip(labels.entity_id, labels.cutoff_time, strict=True):
            admin.run(
                "INSERT INTO __labels VALUES (:e, :c)",
                e=int(entity),
                c=pd.Timestamp(cutoff).to_pydatetime(),
            )
        for label, ir, _, _ in CASES:
            expected = run_feature(
                compile(ir, world["graph"], "duckdb"), world["tables"], labels, world["graph"]
            )
            sql = compile(qualified(ir, f"{SCHEMA}."), graph, "postgres")
            rows = admin.run(sql)
            got = {(int(r[0]), pd.Timestamp(r[1])): r[2] for r in rows}
            assert len(got) == len(labels), label
            for _, row in expected.iterrows():
                key = (int(row["entity_id"]), pd.Timestamp(row["cutoff_time"]))
                assert same(row["value"], got[key]), (label, key, row["value"], got[key])
                compared += 1
    finally:
        admin.close()
    assert compared == 150 * len(CASES)
    print(f"\n{compared} values equal in DuckDB and Postgres across {len(CASES)} IRs")


def test_the_compiled_postgres_sql_qualifies_the_schema(demo_pg) -> None:
    _, ro, _ = demo_pg
    graph = build_schema_graph(open_source(ro))
    ir = qualified(CASES[0][1], f"{SCHEMA}.")
    sql = compile(ir, graph, "postgres")
    assert f"LEFT JOIN {SCHEMA}.refunds t1" in sql
