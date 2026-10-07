# Feature IR

Most useful features of a relational database have one shape: take a column of a related table, aggregate it over a window before the cutoff, with a filter. The feature IR (`backend/ml/features/ir.py`) writes that sentence down as data, and `backend/ml/features/compile.py` turns it into SQL that is point-in-time safe by construction.

It exists because free SQL from a cheap model is messy and hard to prove safe (see [the guard](pit_guard.md)). The IR is small enough for a cheap model to fill in, and it reads back as an English sentence in the report.

The IR is data. Table and column names must exist in the schema graph, and filter values are typed literals that are quoted by sqlglot, so nothing a model writes in it reaches the SQL as text.

## The shape

```yaml
name: refund_count_90d
entity_table: customers
path:                            # edges of the schema graph, from the entity to the source table
  - {from_table: refunds, from_columns: [customer_id], to_table: customers, to_columns: [customer_id]}
source_table: refunds
agg: count                       # count, count_distinct, sum, mean, min, max, std,
                                 # days_since_last, days_since_first, share
column: null                     # null for count, share and days_since_*; required for the others
window_days: 90                  # null = all history before the cutoff
filter:                          # column op value; ops: = != < <= > >= in is_null is_not_null
  - {column: amount, op: ">", value: 20}
ratio_to: null                   # another IR (one level), e.g. 30-day count / 365-day count
transform: none                  # none | log1p
```

`describe(ir)` gives the sentence: *Number of refunds where amount is above 20 in the 90 days before the cutoff*.

## What the compiler emits

```sql
SELECT l.entity_id, l.cutoff_time, count(t1.customer_id) AS value
FROM __labels l
LEFT JOIN refunds t1
  ON t1.customer_id = l.entity_id
  AND t1.refunded_at < l.cutoff_time
  AND t1.refunded_at >= l.cutoff_time - INTERVAL '90 days'
  AND t1.amount > 20
GROUP BY l.entity_id, l.cutoff_time
```

* Every table on the path that has an event time is bounded by `< l.cutoff_time`. The window is a lower bound on the event time of the table nearest the source that has one.
* Filters sit in the join, so an entity with no matching rows still gets a row: `0` for counts and sums, `NULL` for the other aggregates.
* `share` is the share of the rows in the window that match the filter (`avg(CASE WHEN ... THEN 1.0 ELSE 0.0 END)`), `NULL` when there are none. `days_since_*` are fractional days between the cutoff and the latest or earliest event inside the window.
* The path may go from a parent to a child (customers → orders → order_items) or from a child to a parent (… → order_items → products). Rows are counted along the path, so an aggregate over a parent reached through a child counts one row per child row.
* SQL semantics apply to filters: `status != 'x'` does not match rows where `status` is NULL.
* DuckDB and Postgres are supported; `compile(ir, graph, dialect)`. The only differences are the `days_since_*` expression and schema-qualified table names.
* `compile` runs the point-in-time guard on its own output, with rewriting off, and raises if the SQL is not accepted as it is. The guard is a second check that does not trust the compiler.

## Errors

`validate(ir, graph)` returns every problem as `FieldError(field, message)`, with the field path (`column`, `path[1]`, `filter[0].value`, `ratio_to.column`), so a repair prompt can name what to fix. `compile` raises `FeatureIRError` carrying the same list. Checked:

* the column type fits the aggregation (`sum`, `mean`, `min`, `max`, `std` need a numeric column; `count`, `share` and `days_since_*` take no column);
* every path step is an edge of the schema graph, starts where the previous one ended, the first leaves the entity through its key, and the path ends at `source_table`;
* every table on the path has an event time or was confirmed as static (otherwise its rows cannot be placed before the cutoff);
* a window or `days_since_*` needs a table with an event time on the path;
* filter columns exist, operators fit the column type, values match it (numbers for numeric columns, `true`/`false` for booleans, ISO dates for date columns);
* `log1p` only with aggregates that cannot be negative (`count`, `count_distinct`, `std`, `days_since_*`), so it never fails or hides a negative value.

## Limits

* Edges on composite keys are not supported.
* A window is always "the N days before the cutoff"; "the 30 days before the previous 60" is not expressible yet.
* The IR is only compiled and checked here. Nothing fills it in yet (the LLM proposal step is #56) and no run uses it before the baseline features (#55).

## The baseline (DFS)

Before any model proposes a feature, MLPilot builds a baseline with no LLM: `ml/features/dfs.py` writes feature specs for every table one or two edges below the entity table (customers → orders → order_items), and `ml/features/baseline.py` computes them, filters them and trains LightGBM on the temporal split. Every candidate is compiled with the compiler above and checked by the guard, so there is no separate code path for the baseline. It is the bar that proposed features must beat.

Templates, in priority order (the cap keeps the first ones):

| priority | features |
|---|---|
| 0 | count over 7, 30, 90, 365 days and all history |
| 1 | days since the most recent and the first event |
| 2 | count per top-5 value of a low-cardinality text column (30, 90, 365 days); recent share: count in 7 days / count in 30 days, 30 / 90 |
| 3 | sum and mean of numeric columns (30, 90, 365 days); share of rows where a boolean column is true |
| 4 | min, max and standard deviation of numeric columns |
| 5 | columns of the entity row itself, and its age at the cutoff |

Filters, applied in this order to the training rows only (never validation or test):

1. constant, or one value in more than 99% of rows: dropped;
2. equal to, or correlated above 0.99 with, a feature already kept: dropped as a duplicate;
3. more than 300 kept (`max_features`): the rest are dropped as `over_cap`.

Every dropped feature is reported with its reason. A table with no event time that was not confirmed as static, or whose only time column is a last-modified time, is skipped and the reason is reported.

The attributes of the entity row are the one kind of feature the cutoff guard cannot vouch for: a column such as `customers.is_churned` is filled in after the event it describes, and the guard cannot see that (#139). They go through the leakage scan on the training rows (post-outcome names, a copy of the target, one column that predicts almost perfectly), and what it flags is dropped. Text columns with more than 50 distinct values (e-mail addresses, codes) are not used as categories.

LightGBM stops early on the validation rows, so the validation score is slightly optimistic. The test rows are not read; the test set is scored once per run, at its end.

`POST /projects/{id}/tasks/{task}/runs/{run}/baseline` runs it for a run started on a snapshot, and records an experiment (the run's first champion), one `features` row per kept feature (SQL, spec, sentence, gain share) and a summary in the run's manifest. The SQL is compiled for DuckDB and run on the snapshot's copies of the tables; tables are keyed by bare name there.

Limits: the "trend" feature is the recent share of a longer window (7 / 30 days, 30 / 90 days), not a ratio to the previous period, because a window is always measured back from the cutoff. Duplicate detection samples up to 5,000 training rows.
