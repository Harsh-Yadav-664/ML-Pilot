# Demo database

A synthetic e-commerce database for demos, tests and CI: nine tables, about 750,000 rows, two
years of history (2023-01-01 to 2025-01-01), roughly 50 MB. It is generated, not copied from
anyone: `generate.py` is seeded and uses numpy only, so the same seed always gives the same
data (a test compares SHA-256 checksums of every table).

```bash
docker compose -f docker/demo-db/docker-compose.yml up demo-db     # PostgreSQL on 127.0.0.1:5433
psql "postgresql://mlpilot_ro:mlpilot_demo_ro@127.0.0.1:5433/demo" -c '\dt'
```

The first start builds the schema (`schema.sql`), loads the data (`generate.py --copy`) and
creates two roles (`roles.sql`). The passwords above are demo defaults for fake data on
localhost; override them with `DEMO_RO_PASSWORD`, `DEMO_RW_PASSWORD` and `DEMO_ADMIN_PASSWORD`.

| Role | Can do |
|---|---|
| `mlpilot_ro` | `SELECT` only. An `INSERT` fails with `permission denied`. This is the role to connect MLPilot with. |
| `mlpilot_rw` | Also `INSERT`, `UPDATE`, `DELETE`. Connecting with it shows the "this role can write" warning. |

## Without Docker

```bash
python docker/demo-db/generate.py --sqlite demo.sqlite            # SQLite, foreign keys declared
python docker/demo-db/generate.py --duckdb demo.duckdb
python docker/demo-db/generate.py --sqlite demo_nofk.sqlite --no-fks   # same data, no foreign keys
python docker/demo-db/generate.py --checksums                     # row counts and hashes
python docker/demo-db/generate.py --check-signal                  # AUC of a plain recency feature
python docker/demo-db/generate.py --customers 2000 --seed 3 --sqlite small.sqlite
```

The `--no-fks` copy exists to test relationship inference (#45): same tables and data, but nothing
declares how they join.

## Tables

| Table | Rows (default) | Time column | Notes |
|---|---|---|---|
| `customers` | 10,000 | `signup_at` | country, acquisition channel, plan, `email` (a column to keep away from the LLM) |
| `products` | 200 | none (static) | category, price |
| `orders` | ~150,000 | `ordered_at` | status `completed` / `cancelled` / `returned`, total |
| `order_items` | ~255,000 | none (child of `orders`) | quantity, unit price |
| `sessions` | ~136,000 | `started_at` | device, pages |
| `support_tickets` | ~20,000 | `opened_at` | category, `resolved_at` (NULL while open), satisfaction 1-5 |
| `refunds` | ~7,400 | `refunded_at` | amount |
| `marketing_emails` | ~163,000 | `sent_at` | opened |
| `customer_status_snapshot` | 10,000 | `updated_at` only | see the leaks below |

Nine foreign keys: every `customer_id` column to `customers`, `order_items` to `orders` and
`products`, `refunds` to `orders`.

## Behaviour model (so the expected feature importance is known)

- Each customer has a latent **engagement**: a mean-reverting random walk around a personal
  baseline (higher for premium plans and referrals), with rare sudden drops. Orders and sessions
  are Poisson with a rate proportional to engagement and a Nov/Dec seasonal bump. Support tickets
  are more likely when engagement is low, and their satisfaction follows engagement.
- A customer silently **churns** (stops ordering, sessions and tickets) with a weekly hazard that
  rises when engagement is low, for a few weeks after a bad ticket (satisfaction 2 or less) and
  after a refund. About a third of customers churn within the two years.
- So the label "no order in the next 30 days" depends on: recency of the last order, order
  frequency, session counts, ticket satisfaction, refunds and the plan. A plain "days since the
  last order" feature reaches about 0.70 AUC at a cutoff of 2024-06-01 (about 54% of customers with
  an order before that date place none in the next 30 days). `--check-signal` and
  `tests/unit/test_demo_db.py` check that it stays above 0.65.

## Planted traps for the leakage checks (#54)

All three are computed from each customer's final state, so a model that uses them is cheating:

- `customers.is_churned`: set at the end of the simulation.
- `customers.discount_code_used_after_churn`: only churned customers can have one.
- `customer_status_snapshot`: rewritten after churn. Its `status` is `churned` for churned
  customers and its `updated_at` (its only time column) is later than their last order.

## Not included yet

The RelBench `rel-f1` loader for Phase 6 needs a download of that dataset and is left to that phase.
