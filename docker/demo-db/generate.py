"""Synthetic e-commerce database for MLPilot demos, tests and CI (issue #47).

Seeded and numpy-only: the same ``--customers`` and ``--seed`` always give byte-identical
tables. ~10k customers over two years (2023-01-01 to 2025-01-01) in nine tables, about
800k rows and well under 200 MB.

The behaviour model (so the expected feature importance is known, see README.md):

* Each customer has a latent *engagement* that follows a mean-reverting random walk
  with occasional sudden drops. Order, session and support-ticket rates depend on it
  (higher engagement = more orders and sessions, fewer tickets), with Nov/Dec seasonality.
* A customer stops being active ("churns", silently) with a weekly hazard that rises when
  engagement is low, after a bad support ticket (satisfaction <= 2) and after a refund.
* So *recency* (days since the last order), *frequency*, *session counts*, *ticket
  satisfaction* and *refunds* genuinely predict the label "no order in the next 30 days".

Traps for the leakage checks (#54), all computed from the customer's final state:

* ``customers.is_churned`` is set at the end of the simulation.
* ``customers.discount_code_used_after_churn`` only exists for customers who churned.
* ``customer_status_snapshot`` is rewritten after churn: its ``updated_at`` is the churn
  date for churned customers, and it is the only time column of that table.

Usage::

    python generate.py --schema [--no-fks]          # PostgreSQL DDL
    python generate.py --copy                       # the data as COPY ... FROM STDIN blocks
    python generate.py --sqlite out.db [--no-fks]   # SQLite export of the same data
    python generate.py --duckdb out.duckdb [--no-fks]
    python generate.py --checksums                  # row counts and SHA-256 per table
    python generate.py --check-signal               # AUC of a plain recency feature
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CUSTOMERS = 10_000
DEFAULT_SEED = 7
START = np.datetime64("2023-01-01T00:00:00")
END = np.datetime64("2025-01-01T00:00:00")
N_WEEKS = 105
WEEK = 7 * 86_400
HORIZON_S = int((END - START) / np.timedelta64(1, "s"))
SIGNUP_DAYS = 540  # customers sign up during the first ~18 months

COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "CA", "AU"]
CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
PLANS = ["free", "basic", "premium"]
CATEGORIES = ["electronics", "home", "garden", "toys", "books", "beauty", "sports", "grocery"]
TICKET_CATEGORIES = ["billing", "shipping", "product", "account", "other"]
DEVICES = ["mobile", "desktop", "tablet"]


# -- schema: one definition, DDL for every dialect --------------------------------------


@dataclass(frozen=True)
class Col:
    name: str
    type: str  # int | float | text | ts | bool
    nullable: bool = True
    pk: bool = False
    fk: str | None = None  # "table.column"


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Col, ...]


SCHEMA: tuple[Table, ...] = (
    Table(
        "customers",
        (
            Col("customer_id", "int", False, pk=True),
            Col("email", "text"),
            Col("signup_at", "ts", False),
            Col("country", "text"),
            Col("acquisition_channel", "text"),
            Col("plan", "text"),
            Col("is_churned", "bool"),
            Col("discount_code_used_after_churn", "text"),
        ),
    ),
    Table(
        "products",
        (
            Col("product_id", "int", False, pk=True),
            Col("name", "text"),
            Col("category", "text"),
            Col("price", "float"),
        ),
    ),
    Table(
        "orders",
        (
            Col("order_id", "int", False, pk=True),
            Col("customer_id", "int", False, fk="customers.customer_id"),
            Col("ordered_at", "ts", False),
            Col("status", "text"),
            Col("total", "float"),
        ),
    ),
    Table(
        "order_items",
        (
            Col("order_item_id", "int", False, pk=True),
            Col("order_id", "int", False, fk="orders.order_id"),
            Col("product_id", "int", False, fk="products.product_id"),
            Col("quantity", "int"),
            Col("unit_price", "float"),
        ),
    ),
    Table(
        "sessions",
        (
            Col("session_id", "int", False, pk=True),
            Col("customer_id", "int", False, fk="customers.customer_id"),
            Col("started_at", "ts", False),
            Col("device", "text"),
            Col("pages", "int"),
        ),
    ),
    Table(
        "support_tickets",
        (
            Col("ticket_id", "int", False, pk=True),
            Col("customer_id", "int", False, fk="customers.customer_id"),
            Col("opened_at", "ts", False),
            Col("category", "text"),
            Col("resolved_at", "ts"),
            Col("satisfaction", "int"),
        ),
    ),
    Table(
        "refunds",
        (
            Col("refund_id", "int", False, pk=True),
            Col("order_id", "int", False, fk="orders.order_id"),
            Col("customer_id", "int", False, fk="customers.customer_id"),
            Col("refunded_at", "ts", False),
            Col("amount", "float"),
        ),
    ),
    Table(
        "marketing_emails",
        (
            Col("email_id", "int", False, pk=True),
            Col("customer_id", "int", False, fk="customers.customer_id"),
            Col("sent_at", "ts", False),
            Col("opened", "bool"),
        ),
    ),
    Table(
        "customer_status_snapshot",
        (
            Col("customer_id", "int", False, pk=True, fk="customers.customer_id"),
            Col("status", "text"),
            Col("updated_at", "ts", False),
        ),
    ),
)
TABLES = {t.name: t for t in SCHEMA}

SQL_TYPES = {
    "postgres": {
        "int": "bigint",
        "float": "double precision",
        "text": "text",
        "ts": "timestamp",
        "bool": "boolean",
    },
    "sqlite": {
        "int": "INTEGER",
        "float": "REAL",
        "text": "TEXT",
        "ts": "TIMESTAMP",
        "bool": "BOOLEAN",
    },
    "duckdb": {
        "int": "BIGINT",
        "float": "DOUBLE",
        "text": "VARCHAR",
        "ts": "TIMESTAMP",
        "bool": "BOOLEAN",
    },
}


# Indexes a real shop database would have on its foreign keys and event times.
INDEXES = (
    ("orders", ("customer_id", "ordered_at")),
    ("order_items", ("order_id",)),
    ("sessions", ("customer_id", "started_at")),
    ("support_tickets", ("customer_id", "opened_at")),
    ("refunds", ("order_id",)),
    ("marketing_emails", ("customer_id", "sent_at")),
)


def ddl(dialect: str = "postgres", *, fks: bool = True) -> list[str]:
    """CREATE TABLE statements in dependency order, then the indexes."""
    types = SQL_TYPES[dialect]
    out = []
    for table in SCHEMA:
        lines = [
            f"    {c.name} {types[c.type]}{'' if c.nullable else ' NOT NULL'}"
            for c in table.columns
        ]
        pk = [c.name for c in table.columns if c.pk]
        lines.append(f"    PRIMARY KEY ({', '.join(pk)})")
        if fks:
            for c in table.columns:
                if c.fk:
                    ref_table, ref_col = c.fk.split(".")
                    lines.append(f"    FOREIGN KEY ({c.name}) REFERENCES {ref_table} ({ref_col})")
        out.append(f"CREATE TABLE {table.name} (\n" + ",\n".join(lines) + "\n)")
    for table_name, columns in INDEXES:
        out.append(
            f"CREATE INDEX idx_{table_name}_{'_'.join(columns)} ON {table_name} ({', '.join(columns)})"
        )
    return out


def schema_sql(*, fks: bool = True) -> str:
    """The contents of schema.sql (checked against the committed file by a test)."""
    header = (
        "-- Generated by generate.py --schema (the schema is defined there). Do not edit.\n"
        "-- Synthetic e-commerce demo database for MLPilot (issue #47).\n\n"
    )
    return header + ";\n\n".join(ddl("postgres", fks=fks)) + ";\n"


# -- the simulation -------------------------------------------------------------------


def _ts(seconds: np.ndarray) -> np.ndarray:
    """Seconds since START as datetime64[s]."""
    return START + seconds.astype("int64").astype("timedelta64[s]")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_customers: int = DEFAULT_CUSTOMERS, seed: int = DEFAULT_SEED
) -> dict[str, dict[str, Any]]:
    """All nine tables as ``{table: {column: numpy array}}`` (``None``/NaT/NaN = NULL)."""
    rng = np.random.default_rng(seed)
    n = n_customers

    # -- customers --
    signup_s = np.sort(rng.integers(0, SIGNUP_DAYS * 86_400, n))
    signup_week = signup_s // WEEK
    country = rng.choice(COUNTRIES, n, p=[0.35, 0.12, 0.1, 0.08, 0.15, 0.08, 0.07, 0.05])
    channel = rng.choice(CHANNELS, n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    plan = rng.choice(PLANS, n, p=[0.5, 0.35, 0.15])
    base = rng.beta(2.5, 2.5, n)
    base += np.where(plan == "premium", 0.1, 0.0) + np.where(plan == "free", -0.05, 0.0)
    base += np.where(channel == "referral", 0.08, 0.0) + np.where(
        channel == "paid_search", -0.03, 0.0
    )
    base = np.clip(base, 0.08, 0.95)

    # -- products --
    n_products = 200
    prod_category = rng.choice(CATEGORIES, n_products)
    prod_price = np.clip(np.round(rng.lognormal(3.2, 0.7, n_products), 2), 2.0, 500.0)
    prod_weight = 1.0 / np.arange(1, n_products + 1) ** 0.8
    prod_weight /= prod_weight.sum()

    # -- weekly simulation --
    e = base.copy()
    churned = np.zeros(n, dtype=bool)
    churn_week = np.full(n, -1, dtype=np.int64)
    bad_recent = np.zeros(n)
    refund_recent = np.zeros(n)
    cust_idx = np.arange(n)

    o_cust: list[np.ndarray] = []
    o_time: list[np.ndarray] = []
    o_status: list[np.ndarray] = []
    s_cust: list[np.ndarray] = []
    s_time: list[np.ndarray] = []
    t_cust: list[np.ndarray] = []
    t_time: list[np.ndarray] = []
    t_sat: list[np.ndarray] = []
    m_cust: list[np.ndarray] = []
    m_time: list[np.ndarray] = []
    m_open: list[np.ndarray] = []

    for w in range(N_WEEKS):
        week_start = w * WEEK
        mid_day = (week_start + WEEK // 2) / 86_400.0
        season = 1.0 + 0.3 * np.cos(2 * np.pi * ((mid_day % 365.0) - 340.0) / 365.0)
        started = w > signup_week
        active = started & ~churned

        # engagement: mean-reverting walk, with rare sudden drops
        e = 0.92 * e + 0.08 * base + rng.normal(0, 0.05, n)
        shock = active & (rng.random(n) < 0.008)
        e = np.clip(np.where(shock, e - 0.3, e), 0.02, 1.0)

        def _times(counts: np.ndarray, start: int = week_start) -> np.ndarray:
            return start + rng.integers(0, WEEK, int(counts.sum()))

        # orders
        n_orders = rng.poisson(0.55 * e * season * active)
        o_c = np.repeat(cust_idx, n_orders)
        status = rng.choice(["completed", "cancelled", "returned"], len(o_c), p=[0.9, 0.04, 0.06])
        o_cust.append(o_c)
        o_time.append(_times(n_orders))
        o_status.append(status)
        returned = np.bincount(o_c[status == "returned"], minlength=n)

        # sessions
        n_sess = rng.poisson(0.5 * e * season * active)
        s_cust.append(np.repeat(cust_idx, n_sess))
        s_time.append(_times(n_sess))

        # support tickets: more when engagement is low; satisfaction follows engagement
        n_tick = rng.poisson((0.015 + 0.05 * (1 - e)) * active)
        t_c = np.repeat(cust_idx, n_tick)
        sat = np.clip(np.rint(rng.normal(2.0 + 2.6 * e[t_c], 0.9)), 1, 5).astype(np.int64)
        t_cust.append(t_c)
        t_time.append(_times(n_tick))
        t_sat.append(sat)
        bad_new = np.bincount(t_c[sat <= 2], minlength=n)

        # weekly marketing email to most signed-up customers; churned ones barely open
        sent = started & (rng.random(n) < 0.25)
        m_c = cust_idx[sent]
        open_p = 0.04 + 0.55 * e[m_c] * np.where(churned[m_c], 0.15, 1.0)
        m_cust.append(m_c)
        m_time.append(week_start + rng.integers(0, WEEK, len(m_c)))
        m_open.append(rng.random(len(m_c)) < open_p)

        # bad experiences raise the churn hazard for the next few weeks
        bad_recent = 0.8 * bad_recent + bad_new
        refund_recent = 0.8 * refund_recent + 0.85 * returned
        hazard = _sigmoid(
            -5.6
            + 6.0 * (0.45 - e)
            + 0.9 * np.minimum(bad_recent, 2.0)
            + 0.7 * np.minimum(refund_recent, 2.0)
        )
        newly = active & (rng.random(n) < hazard)
        churn_week[newly] = w + 1
        churned |= newly

    # -- orders, items, refunds --
    order_cust = np.concatenate(o_cust)
    order_s = np.concatenate(o_time)
    order_status = np.concatenate(o_status)
    keep = order_s < HORIZON_S
    order_cust, order_s, order_status = order_cust[keep], order_s[keep], order_status[keep]
    by_time = np.argsort(order_s, kind="stable")
    order_cust, order_s, order_status = order_cust[by_time], order_s[by_time], order_status[by_time]
    m = len(order_cust)
    order_id = np.arange(1, m + 1)

    n_items = np.clip(1 + rng.poisson(0.7, m), 1, 5)
    item_order = np.repeat(order_id, n_items)
    item_product = rng.choice(np.arange(1, n_products + 1), len(item_order), p=prod_weight)
    item_qty = rng.choice([1, 2, 3], len(item_order), p=[0.75, 0.18, 0.07])
    discount = np.where(rng.random(len(item_order)) < 0.1, 0.9, 1.0)
    item_price = np.round(prod_price[item_product - 1] * discount, 2)
    order_total = np.zeros(m)
    np.add.at(order_total, item_order - 1, item_qty * item_price)

    returned_ids = order_id[order_status == "returned"]
    refunded = returned_ids[rng.random(len(returned_ids)) < 0.85]
    refund_s = order_s[refunded - 1] + (rng.uniform(2, 21, len(refunded)) * 86_400).astype(np.int64)
    in_horizon = refund_s < HORIZON_S
    refunded, refund_s = refunded[in_horizon], refund_s[in_horizon]
    refund_amount = np.round(order_total[refunded - 1] * rng.uniform(0.6, 1.0, len(refunded)), 2)

    # -- sessions, tickets, emails (cut at END, ids in time order) --
    def _stack(cust: list[np.ndarray], secs: list[np.ndarray], *extra: list[np.ndarray]):
        c, s = np.concatenate(cust), np.concatenate(secs)
        ex = [np.concatenate(x) for x in extra]
        ok = s < HORIZON_S
        order = np.argsort(s[ok], kind="stable")
        return (c[ok][order], s[ok][order], *[x[ok][order] for x in ex])

    sess_cust, sess_s = _stack(s_cust, s_time)
    sess_device = rng.choice(DEVICES, len(sess_cust), p=[0.55, 0.4, 0.05])
    sess_pages = 1 + rng.poisson(3 + 6 * e[sess_cust], len(sess_cust))
    tick_cust, tick_s, tick_sat = _stack(t_cust, t_time, t_sat)
    tick_category = rng.choice(TICKET_CATEGORIES, len(tick_cust), p=[0.25, 0.3, 0.2, 0.15, 0.1])
    has_resolution = rng.random(len(tick_cust)) < 0.9
    resolved_s = tick_s + (rng.exponential(2.0, len(tick_cust)) * 86_400).astype(np.int64)
    resolved = np.where(
        has_resolution & (resolved_s < HORIZON_S), _ts(resolved_s), np.datetime64("NaT", "s")
    )
    mail_cust, mail_s, mail_open = _stack(m_cust, m_time, m_open)

    # -- customer columns that depend on the final state (the planted leaks) --
    is_churned = churn_week >= 0
    churn_s = np.minimum(churn_week * WEEK, HORIZON_S - 1)
    discount_code = np.where(
        is_churned & (rng.random(n) < 0.6),
        np.char.add("WINBACK", rng.integers(10, 99, n).astype(str)),
        None,
    ).astype(object)
    snapshot_s = np.where(
        is_churned,
        np.minimum(churn_s + rng.integers(0, 30 * 86_400, n), HORIZON_S - 1),
        rng.integers(signup_s, HORIZON_S),
    )
    snapshot_status = np.where(is_churned, "churned", np.where(e < 0.25, "at_risk", "active"))
    customer_id = np.arange(1, n + 1)

    def _ids(k: int) -> np.ndarray:
        return np.arange(1, k + 1)

    return {
        "customers": {
            "customer_id": customer_id,
            "email": np.char.add(
                np.char.add("customer", customer_id.astype(str)), "@example.com"
            ).astype(object),
            "signup_at": _ts(signup_s),
            "country": country.astype(object),
            "acquisition_channel": channel.astype(object),
            "plan": plan.astype(object),
            "is_churned": is_churned,
            "discount_code_used_after_churn": discount_code,
        },
        "products": {
            "product_id": _ids(n_products),
            "name": np.char.add("Product ", _ids(n_products).astype(str)).astype(object),
            "category": prod_category.astype(object),
            "price": prod_price,
        },
        "orders": {
            "order_id": order_id,
            "customer_id": order_cust + 1,
            "ordered_at": _ts(order_s),
            "status": order_status.astype(object),
            "total": np.round(order_total, 2),
        },
        "order_items": {
            "order_item_id": _ids(len(item_order)),
            "order_id": item_order,
            "product_id": item_product,
            "quantity": item_qty,
            "unit_price": item_price,
        },
        "sessions": {
            "session_id": _ids(len(sess_cust)),
            "customer_id": sess_cust + 1,
            "started_at": _ts(sess_s),
            "device": sess_device.astype(object),
            "pages": sess_pages,
        },
        "support_tickets": {
            "ticket_id": _ids(len(tick_cust)),
            "customer_id": tick_cust + 1,
            "opened_at": _ts(tick_s),
            "category": tick_category.astype(object),
            "resolved_at": resolved,
            "satisfaction": tick_sat,
        },
        "refunds": {
            "refund_id": _ids(len(refunded)),
            "order_id": refunded,
            "customer_id": order_cust[refunded - 1] + 1,
            "refunded_at": _ts(refund_s),
            "amount": refund_amount,
        },
        "marketing_emails": {
            "email_id": _ids(len(mail_cust)),
            "customer_id": mail_cust + 1,
            "sent_at": _ts(mail_s),
            "opened": mail_open,
        },
        "customer_status_snapshot": {
            "customer_id": customer_id,
            "status": snapshot_status.astype(object),
            "updated_at": _ts(snapshot_s),
        },
    }


# -- serialisation: Postgres COPY text, SQLite and DuckDB rows ----------------------------


def _column_text(col: Col, values: np.ndarray) -> list[str]:
    """One column as COPY text values (``\\N`` for NULL)."""
    if col.type == "ts":
        text = np.datetime_as_string(values, unit="s")
        return [r"\N" if s == "NaT" else s.replace("T", " ") for s in text.tolist()]
    if col.type == "bool":
        return ["t" if v else "f" for v in values.tolist()]
    if col.type == "float":
        return [r"\N" if np.isnan(v) else repr(float(v)) for v in values.tolist()]
    if col.type == "int":
        return [str(int(v)) for v in values.tolist()]
    out = []
    for v in values.tolist():
        if v is None:
            out.append(r"\N")
        else:
            out.append(str(v).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n"))
    return out


def table_text(data: dict[str, dict[str, Any]], name: str) -> str:
    """The table as PostgreSQL COPY text (tab-separated, one row per line)."""
    table = TABLES[name]
    columns = [_column_text(c, data[name][c.name]) for c in table.columns]
    return "".join("\t".join(row) + "\n" for row in zip(*columns, strict=True))


def copy_blocks(data: dict[str, dict[str, Any]]) -> Iterator[str]:
    for table in SCHEMA:
        cols = ", ".join(c.name for c in table.columns)
        yield f"COPY {table.name} ({cols}) FROM STDIN;\n"
        yield table_text(data, table.name)
        yield "\\.\n\n"


def checksums(data: dict[str, dict[str, Any]]) -> dict[str, tuple[int, str]]:
    """``{table: (row count, sha256 of its COPY text)}``."""
    out = {}
    for table in SCHEMA:
        text = table_text(data, table.name)
        out[table.name] = (text.count("\n"), hashlib.sha256(text.encode()).hexdigest())
    return out


def _python_rows(
    data: dict[str, dict[str, Any]], name: str, *, ts_as: str
) -> list[tuple[Any, ...]]:
    table = TABLES[name]
    columns: list[list[Any]] = []
    for c in table.columns:
        values = data[name][c.name]
        if c.type == "ts":
            text = np.datetime_as_string(values, unit="s").tolist()
            columns.append([None if s == "NaT" else s.replace("T", " ") for s in text])
        elif c.type == "bool":
            columns.append([bool(v) if ts_as == "duckdb" else int(v) for v in values.tolist()])
        elif c.type == "float":
            columns.append([None if np.isnan(v) else float(v) for v in values.tolist()])
        elif c.type == "int":
            columns.append([int(v) for v in values.tolist()])
        else:
            columns.append(values.tolist())
    return list(zip(*columns, strict=True))


def write_sqlite(data: dict[str, dict[str, Any]], path: str | Path, *, fks: bool = True) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    try:
        for statement in ddl("sqlite", fks=fks):
            con.execute(statement)
        for table in SCHEMA:
            marks = ", ".join("?" * len(table.columns))
            con.executemany(
                f"INSERT INTO {table.name} VALUES ({marks})",
                _python_rows(data, table.name, ts_as="sqlite"),
            )
        con.commit()
    finally:
        con.close()


def write_duckdb(data: dict[str, dict[str, Any]], path: str | Path, *, fks: bool = True) -> None:
    import duckdb
    import pandas as pd

    path = Path(path)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    try:
        for statement in ddl("duckdb", fks=fks):
            con.execute(statement)
        for table in SCHEMA:
            names = [c.name for c in table.columns]
            frame = pd.DataFrame(_python_rows(data, table.name, ts_as="duckdb"), columns=names)
            for c in table.columns:
                if c.type == "ts":
                    frame[c.name] = pd.to_datetime(frame[c.name])
            con.register("incoming", frame)
            con.execute(f"INSERT INTO {table.name} SELECT * FROM incoming")
            con.unregister("incoming")
    finally:
        con.close()


def load_postgres(conn: Any, data: dict[str, dict[str, Any]], *, fks: bool = True) -> None:
    """Create the tables and COPY the data through a pg8000 native connection.

    The caller picks the schema first (``SET search_path``). Existing demo tables must be dropped.
    """
    for statement in ddl("postgres", fks=fks):
        conn.run(statement)
    for table in SCHEMA:
        cols = ", ".join(c.name for c in table.columns)
        conn.run(
            f"COPY {table.name} ({cols}) FROM STDIN",
            stream=io.BytesIO(table_text(data, table.name).encode()),
        )


# -- the signal check: a plain recency feature must predict the label ---------------------


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC by the rank-sum (Mann-Whitney) formula, ties averaged."""
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


RECENCY_SQL = """
SELECT c.customer_id,
       MAX(o.ordered_at) AS last_order,
       (SELECT COUNT(*) FROM orders n
         WHERE n.customer_id = c.customer_id AND n.ordered_at > ? AND n.ordered_at <= ?) AS next_30d
FROM customers c JOIN orders o ON o.customer_id = c.customer_id AND o.ordered_at <= ?
GROUP BY c.customer_id
"""


def recency_auc(
    con: sqlite3.Connection, cutoff: str = "2024-06-01 00:00:00"
) -> tuple[float, int, float]:
    """AUC of "days since last order" for the label "no order in the next 30 days".

    Returns ``(auc, customers scored, share with the label)``; ``cutoff`` is a ``YYYY-MM-DD
    HH:MM:SS`` string. Runs on the SQLite export.
    """
    end = str(np.datetime64(cutoff) + np.timedelta64(30, "D")).replace("T", " ")
    rows = con.execute(RECENCY_SQL, (cutoff, end, cutoff)).fetchall()
    cut = np.datetime64(cutoff)
    days = np.array([(cut - np.datetime64(r[1])) / np.timedelta64(1, "D") for r in rows])
    label = np.array([1 if r[2] == 0 else 0 for r in rows])
    return rank_auc(days, label), len(rows), float(label.mean())


# -- command line -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-fks", action="store_true", help="declare no foreign keys")
    parser.add_argument("--schema", action="store_true", help="print the PostgreSQL DDL")
    parser.add_argument("--copy", action="store_true", help="print the data as COPY blocks")
    parser.add_argument("--sqlite", metavar="PATH")
    parser.add_argument("--duckdb", metavar="PATH")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--check-signal", action="store_true")
    args = parser.parse_args(argv)
    fks = not args.no_fks

    if args.schema:
        sys.stdout.write(schema_sql(fks=fks))
        return 0
    if not (args.copy or args.sqlite or args.duckdb or args.checksums or args.check_signal):
        parser.error(
            "nothing to do: pass --schema, --copy, --sqlite, --duckdb, --checksums or --check-signal"
        )

    data = generate(args.customers, args.seed)
    if args.copy:
        for block in copy_blocks(data):
            sys.stdout.write(block)
    if args.sqlite:
        write_sqlite(data, args.sqlite, fks=fks)
    if args.duckdb:
        write_duckdb(data, args.duckdb, fks=fks)
    if args.checksums:
        for name, (rows, digest) in checksums(data).items():
            print(f"{name:26} {rows:>9} rows  sha256 {digest[:16]}")
    if args.check_signal:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "demo.sqlite"
            write_sqlite(data, db)
            con = sqlite3.connect(db)
            try:
                auc, n_scored, churn_share = recency_auc(con)
            finally:
                con.close()
        print(
            f"recency (days since last order) vs 'no order in the next 30 days': "
            f"AUC {auc:.3f} on {n_scored} customers, {churn_share:.1%} positive"
        )
        if auc <= 0.65:
            print("FAIL: AUC must be above 0.65", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
