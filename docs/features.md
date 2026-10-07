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

## LLM-proposed features

`ml/features/llm_sql.py` asks the model for one feature at a time. The prompt (built by the context builder, purpose `features.propose`) holds the task in words, the schema and column statistics, the strongest features already in use in plain English (per-category counts are described without their values), the names taken, the proposals rejected so far with the reason, and the JSON Schema of an answer. An answer is either an `ir` (the Feature IR above) or, if the project turned free SQL on, a `sql` string. No cell values are sent.

Every answer goes through the same chain, and the first failing check stops it:

1. **schema**: the answer parses, the name is new, the entity table is the task's, the IR validates against the schema graph (free SQL is refused unless enabled)
2. **guard**: the IR compiles, or the free SQL passes the point-in-time guard
3. **duplicate**: the same spec or the same normalised query is already in use
4. **execution**: the query runs on the snapshot, keeps the label row contract, and is not constant on the training rows
5. **duplicate_after_execution**: the values are not (almost) a copy of a feature in use (correlation above 0.98)

A rejected answer is sent back once, with the stage and reasons; a second rejection is final. There is no third attempt. A proposal that passes everything is `proposed`: whether it helps is decided later, by validated gain (#58), never by the model.

If no real model answers (the offline stub, or every provider failing), the result is `no_llm`: nothing is invented and the baseline stands. The tests use a scripted model (`tests/fixtures/stub_feature_proposals.yaml`) that stands for what a model might answer; it is a test double, not the stub provider.

`app/services/feature_proposals.py` stores each proposal as a `Feature` row (`kind: llm_sql`) with the stage, reasons and attempts in `guard_results`. How many proposals from a real model pass is measured by `python -m scripts.feature_proposal_rate` (CI job `feature-proposals`, runs only with an API key secret); that number has not been measured yet.

## Running features: `ml/features/engine.py`

`FeatureEngine` loads the tables of a data version (a snapshot) and the label rows into one in-memory DuckDB, then runs a feature as **one query over every label row**, not one per row. Its values are cached as Parquet under a key made of the normalised query (the same query in other words hits the cache), the data version id, the label table version and the entity sample. A second run on the same data and labels reads the values back instead of computing them; a new data version or label version computes them again. A failed feature is never cached.

- **Time limit:** each query has a limit (default 120 seconds). A query over it is interrupted and returned as `failed` with a reason starting `timeout:`; a result that breaks the `entity_id, cutoff_time, value` contract is `failed` with `contract:`. The engine stays usable and the caller goes on with the next feature. The proposer records such a feature as a rejected proposal with that reason.
- **Sampling:** if entities x cutoffs is above `max_rows` (default 2,000,000) a fixed-seed sample of entities is kept, stratified by whether the entity ever has a positive label, and every cutoff of a kept entity is kept. The fraction, the seed and the counts are in `engine.sampling` so they can be recorded and shown. The model can be refit on the full table afterwards; that refit is not built yet.
- **Budgets:** `Budget(max_cost_usd, max_seconds, max_proposals)` and `BudgetTracker`. `propose_within_budget` checks before each proposal and charges after it, so the step in progress always finishes and a run can go over a limit by at most one proposal. When a limit is reached the run is recorded as `stopped` (`runs.status`), with the limits and what was used in `runs.budget` / `runs.budget_used`, a message in `runs.error`, and the champion untouched. Progress and the stop are emitted as job events (`feature_proposal`, `budget_stop`).

The queries run in a DuckDB with no file or network access, but the engine does not read the SQL: it expects queries that already passed the point-in-time guard (the proposer and the baseline both do that first).

Not built yet: running against a live Postgres through the SQL guard (snapshots only), the refit on the full table, and the run loop that ties the baseline, the proposer and the budget together (#58).

## The run loop: `ml/agents/run_loop.py`

`POST /projects/{id}/runs/{run}/start` queues a job (kind `relational_run`, cancellable, with events) for a run made on a snapshot. The job builds the baseline, records it as the first champion, then repeats up to `max_rounds` times:

1. stop if a budget is reached, or `patience` rounds in a row had no accepted feature
2. ask the proposer for one feature; it is checked, executed and de-duplicated as above
3. score the champion and the champion plus the feature on the **same time-ordered folds of the training rows** (`TemporalSplit.folds`: each fold trains on earlier cutoffs and validates on a later block, with the gap that keeps label windows from overlapping), fitting a smaller LightGBM per fold
4. accept in code if the mean paired PR-AUC gain is larger than `max(min_gain, std_multiplier x std of the paired gains)` (ADR 0006; defaults 0.002 and 1.0). The model is not asked. A rejected feature is remembered, with the reason, in the next prompt
5. an accepted feature becomes the new champion: LightGBM is fitted on the training rows with early stopping on the validation rows, and recorded as an experiment whose parent is the previous champion

At the end, the champion is refit on the training and validation rows and **the test rows are scored once**, by one function (`score_test`) that a test spies on. A run that is cancelled is never scored. A run stopped by a budget (`stopped`) is scored, since it has a champion. Without a real model nothing is proposed (`no_llm`) and the baseline is the result.

What is recorded: every proposal is a `Feature` row (`llm_sql`, with its SQL, the stage that stopped it or its paired gain, `accepted` or `rejected_gain`); each accepted feature and the final model are `Experiment` rows; `GET /runs/{run}` gives the status, stop reason, budget used, validation metrics of the champion and the test metrics; `GET /runs/{run}/features` lists the features.

Not built yet: several independent proposal histories (`rollouts`), resuming an interrupted run from its last champion (the champion is on record after every accepted feature, but nothing restarts from it), the same loop for single-table CSV tasks (they keep the older formula loop), and a UI for a run (#101).

