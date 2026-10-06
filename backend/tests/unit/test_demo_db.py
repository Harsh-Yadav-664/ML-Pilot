"""The demo database generator (#47): deterministic, consistent, with signal and planted leaks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import numpy as np
import pytest

from tests.fixtures.demo_db import GENERATOR, demo_data, demo_duckdb, demo_module, demo_sqlite

DEMO_DIR = GENERATOR.parent
SQL_FKS = {  # table -> {column: referenced table}
    "orders": {"customer_id": "customers"},
    "order_items": {"order_id": "orders", "product_id": "products"},
    "sessions": {"customer_id": "customers"},
    "support_tickets": {"customer_id": "customers"},
    "refunds": {"order_id": "orders", "customer_id": "customers"},
    "marketing_emails": {"customer_id": "customers"},
    "customer_status_snapshot": {"customer_id": "customers"},
}


def test_the_same_seed_gives_identical_table_checksums_and_another_seed_does_not() -> None:
    gen = demo_module()
    first = gen.checksums(gen.generate(300, seed=3))
    again = gen.checksums(gen.generate(300, seed=3))
    other = gen.checksums(gen.generate(300, seed=4))
    assert first == again
    assert set(first) == {t.name for t in gen.SCHEMA} and len(first) == 9
    assert first["customers"] != other["customers"] and first["orders"] != other["orders"]
    print("\n" + "\n".join(f"{t:26} {n:>6} rows  {h[:16]}" for t, (n, h) in first.items()))


def test_the_committed_schema_sql_is_what_the_generator_produces() -> None:
    assert (DEMO_DIR / "schema.sql").read_text() == demo_module().schema_sql()


def test_every_foreign_key_is_declared_and_the_data_respects_them(tmp_path: Path) -> None:
    db = demo_sqlite(tmp_path / "demo.sqlite")
    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        declared = {
            t: {r[3]: r[2] for r in con.execute(f"PRAGMA foreign_key_list({t})")} for t in SQL_FKS
        }
    finally:
        con.close()
    assert declared == SQL_FKS


def test_the_no_fks_copy_has_the_same_data_and_no_foreign_keys(tmp_path: Path) -> None:
    with_fks = demo_sqlite(tmp_path / "a.sqlite")
    without = demo_sqlite(tmp_path / "b.sqlite", fks=False)
    con = sqlite3.connect(without)
    try:
        assert all(con.execute(f"PRAGMA foreign_key_list({t})").fetchall() == [] for t in SQL_FKS)
        ref = sqlite3.connect(with_fks)
        try:
            for table in demo_module().TABLES:
                query = f"SELECT * FROM {table} ORDER BY 1"
                assert con.execute(query).fetchall() == ref.execute(query).fetchall()
        finally:
            ref.close()
    finally:
        con.close()


def test_the_duckdb_export_declares_the_same_foreign_keys(tmp_path: Path) -> None:
    db = demo_duckdb(tmp_path / "demo.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name, constraint_column_names, referenced_table "
            "FROM duckdb_constraints() WHERE constraint_type = 'FOREIGN KEY'"
        ).fetchall()
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in SQL_FKS}
    finally:
        con.close()
    declared: dict[str, dict[str, str]] = {}
    for table, cols, ref in rows:
        declared.setdefault(table, {})[cols[0]] = ref
    assert declared == SQL_FKS
    assert all(n > 0 for n in counts.values())


def test_times_are_consistent_and_inside_the_two_year_window() -> None:
    d = demo_data()
    gen = demo_module()
    for table, column in (
        ("customers", "signup_at"),
        ("orders", "ordered_at"),
        ("sessions", "started_at"),
        ("support_tickets", "opened_at"),
        ("refunds", "refunded_at"),
        ("marketing_emails", "sent_at"),
        ("customer_status_snapshot", "updated_at"),
    ):
        values = d[table][column]
        assert values.min() >= gen.START and values.max() < gen.END, (table, column)
    signup = d["customers"]["signup_at"]
    for table, column in (("orders", "ordered_at"), ("sessions", "started_at")):
        owner = d[table]["customer_id"] - 1
        assert (d[table][column] > signup[owner]).all(), table
    refunds = d["refunds"]
    ordered = d["orders"]["ordered_at"][refunds["order_id"] - 1]
    assert (refunds["refunded_at"] > ordered).all()
    tickets = d["support_tickets"]
    resolved = tickets["resolved_at"]
    assert (resolved[~np.isnat(resolved)] >= tickets["opened_at"][~np.isnat(resolved)]).all()
    assert np.isnat(resolved).any()  # some tickets are still open: real NULLs
    assert (np.diff(d["orders"]["ordered_at"].astype("int64")) >= 0).all()  # ids follow time


def test_order_totals_add_up_and_the_status_mix_is_plausible() -> None:
    d = demo_data()
    items = d["order_items"]
    totals = np.zeros(len(d["orders"]["order_id"]))
    np.add.at(totals, items["order_id"] - 1, items["quantity"] * items["unit_price"])
    assert np.allclose(totals, d["orders"]["total"], atol=0.01)
    status, counts = np.unique(d["orders"]["status"].astype(str), return_counts=True)
    share = dict(zip(status.tolist(), (counts / counts.sum()).tolist(), strict=True))
    assert 0.85 < share["completed"] < 0.95 and share["returned"] > 0.02
    returned = set(d["orders"]["order_id"][d["orders"]["status"] == "returned"].tolist())
    assert set(d["refunds"]["order_id"].tolist()) <= returned


def test_the_planted_leaks_are_there_and_computed_from_the_final_state() -> None:
    d = demo_data()
    customers, snap = d["customers"], d["customer_status_snapshot"]
    churned = customers["is_churned"]
    assert 0.2 < churned.mean() < 0.6
    codes = customers["discount_code_used_after_churn"]
    has_code = np.array([c is not None for c in codes])
    assert has_code.any() and not has_code[~churned].any()  # only churned customers have one
    assert (snap["status"][churned] == "churned").all()
    assert (snap["status"][~churned] != "churned").all()
    # the snapshot was rewritten after churn: later than every order of that customer
    last_order = np.full(len(churned), np.datetime64("1970-01-01T00:00:00"), dtype="datetime64[s]")
    owner = d["orders"]["customer_id"] - 1
    np.maximum.at(last_order, owner, d["orders"]["ordered_at"])
    assert (snap["updated_at"][churned] > last_order[churned]).all()
    assert len(set(snap["customer_id"].tolist())) == len(customers["customer_id"])


def test_a_plain_recency_feature_predicts_who_stops_ordering(tmp_path: Path) -> None:
    """There is signal to find: days since the last order reaches AUC > 0.65 for the label
    'no order in the next 30 days' (3,000 customers, computed from the SQLite export)."""
    gen = demo_module()
    db = tmp_path / "signal.sqlite"
    gen.write_sqlite(gen.generate(3000, 7), db)
    con = sqlite3.connect(db)
    try:
        auc, scored, positive = gen.recency_auc(con)
    finally:
        con.close()
    print(f"\nrecency AUC {auc:.3f} on {scored} customers, {positive:.1%} stop ordering")
    assert auc > 0.65 and scored > 1000 and 0.2 < positive < 0.8


def test_rank_auc_matches_the_definition() -> None:
    gen = demo_module()
    assert gen.rank_auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == 1.0
    assert gen.rank_auc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) == 0.0
    assert gen.rank_auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 1])) == 0.5
    score, label = np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])
    assert gen.rank_auc(score, label) == pytest.approx(0.75)  # sklearn's textbook example


def test_the_default_size_stays_well_under_200_mb(tmp_path: Path) -> None:
    gen = demo_module()
    data = gen.generate()
    rows = sum(n for n, _ in gen.checksums(data).values())
    db = tmp_path / "default.sqlite"
    gen.write_sqlite(data, db)
    size_mb = db.stat().st_size / 1e6
    print(
        f"\ndefault size: {len(data['customers']['customer_id'])} customers, {rows} rows, "
        f"SQLite file {size_mb:.0f} MB"
    )
    assert len(data["customers"]["customer_id"]) == 10_000 and rows > 500_000
    assert size_mb < 200
