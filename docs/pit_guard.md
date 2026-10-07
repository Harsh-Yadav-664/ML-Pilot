# Point-in-time guard

A feature must only use rows from before the cutoff it is computed for. MLPilot enforces that by reading the feature's SQL, not by trusting whoever wrote it (a person, the feature search, or an LLM). The code is `backend/ml/tasks/pit_guard.py`; a second, independent check of the result is `backend/ml/tasks/pit_verify.py`.

## The contract of a feature query

The query reads a relation `__labels(entity_id, cutoff_time)` (one row per label row, from the label builder) and returns exactly `entity_id, cutoff_time, value`, in that order, with at most one row per label row. A missing value is NULL. `cutoff_time` in the output must be `__labels.cutoff_time` itself.

```sql
SELECT l.entity_id, l.cutoff_time, count(o.order_id) AS value
FROM __labels l
LEFT JOIN orders o ON o.customer_id = l.entity_id AND o.ordered_at < l.cutoff_time
GROUP BY l.entity_id, l.cutoff_time
```

## What `check(sql, schema_graph)` returns

| status | meaning |
|---|---|
| `accepted` | Every table with an event time is bounded by the cutoff. |
| `rewritten` | A bound was missing and could be added safely. The returned SQL has it, and `rewrites` says where. |
| `rejected` | It could read the future, or the guard cannot tell. `reasons` name the scope, the table and a stable code. No SQL is returned. |

`assumptions` lists static tables that were read without a time bound. `warnings` lists static assumptions that look wrong (a table marked static that has a last-modified column). Both appear in the evidence report.

## The rules

1. The query is one read-only `SELECT` (it goes through the SQL guard first), is parsed with sqlglot and every column is qualified, so each reference names its table.
2. **Every table with an event time needs a bound in its own scope.** A top-level `AND` term of that scope's `WHERE`, or of the table's own `JOIN ... ON`, of the form `t.time < K`, or `t.time <= K - INTERVAL 'n ...'` with the interval at least one microsecond (a smaller one rounds to zero in the database). `K` is the cutoff. Bounds inside an `OR`, bounds on another table, `<=` the cutoff itself, comparisons on a `DATE` cast, casts with a precision such as `TIMESTAMP(0)` and a column compared with itself do not count. A date-typed event column has no time of day, so it is compared in whole days: `t.d < CAST(K AS DATE)`.
3. **A table without an event time** is accepted only if the user confirmed it as static in the schema. Otherwise the guard says so and names the table.
4. **Which cutoff.** `K` must be `__labels.cutoff_time` or a pass-through of it. A scope's cutoff sources are its own `__labels`, derived tables and CTEs that expose a cutoff column, and, for a correlated subquery, the enclosing row. They must be joined to each other on `cutoff_time` equality (in the `WHERE`, or in the `ON` of one of the two joins involved). Otherwise a derived table could hand over values that were computed for another cutoff.
5. **No mixing of label rows.** In a scope that reads its own cutoff, aggregates must `GROUP BY` the cutoff, `DISTINCT` must keep it in the output, window functions must `PARTITION BY` it, and `LIMIT` is refused (a `LIMIT` in a correlated subquery acts per outer row and is fine). A derived table or CTE that reads `__labels` must expose `cutoff_time`.
6. **Refused outright:** unknown tables and label tables; non-deterministic functions (`now()`, `current_date`, `random()`, `uuid()`, literals such as `'now'`) and any function the SQL parser does not know by name (it could be one of them); order-dependent picks (`any_value`, `first`, `last`, `array_agg` and `string_agg` without `ORDER BY`, `LIMIT` without `ORDER BY`); window frames with `FOLLOWING`; `UNION`/`INTERSECT`/`EXCEPT`; lateral joins, `UNNEST`; every join except `JOIN`, `INNER JOIN`, `CROSS JOIN` and `LEFT JOIN` with `ON` (so no `SEMI`, `ANTI`, `ASOF`, `POSITIONAL`, `NATURAL`, `USING`, right or full joins); `GROUPING SETS`/`ROLLUP`/`CUBE`; `DISTINCT ON`; a second `__labels` in one scope; two output columns with one name; aliases that differ only in case.
7. **Asking for the future is refused by name** (`reads_future`). A term that selects the table's event time from after the cutoff (`t > K`, `t >= K`, `t < K + INTERVAL ...`, `BETWEEN K AND K + INTERVAL ...`, or the same with the column on the right) is not rewritten into `t < K`, which would be safe but would silently turn the feature into an empty one: the query is rejected and the message says a feature may only read history. A lookback such as `t >= K - INTERVAL '90 days'` is fine.
8. **Last-modified times are refused** (`last_modified_time`). If the only time column of a table is a last-modified time (`updated_at`, `modified_at`, ...; the schema graph's `time_leakage_hint`), the table is rejected even with a correct bound: a row is rewritten after the event it describes, so "this row was updated before the cutoff" tells the feature that the future did not happen. If the table is an append-only log (each change is a new row, as in a `plan_changes` table with a `changed_at`), set that column as its event time in the schema: the choice is yours, the guard accepts it and records a warning. If its rows never change, mark it static. Marking a table static that has such a column is accepted but produces the warning "static assumption on a table that changes".
9. **Rewriting.** When a bound is missing and the scope has a cutoff to use, the guard adds `t.time < <cutoff>`: to the table's own `ON` when it is joined, otherwise to the `WHERE` (never when an outer join would then drop label rows). The cutoff it names must resolve, from that scope, to the cutoff it picked (a name shadowed by an inner alias is refused), and inside an `ON` it must belong to a source joined earlier. Anything else that is wrong is rejected, not repaired.

Everything the guard does not model is rejected. The test fixture `backend/tests/fixtures/pit_queries.yaml` lists 79 queries with their expected result.

## The runtime check

`pit_verify.truncation_check` recomputes a feature on a copy of the data in which every row at or after the cutoff has been deleted, for a sample of label rows at several cutoffs, and compares the values. A difference is a leak the SQL analysis missed and is reported with the entity, the cutoff and both values. It also enforces the contract (columns, unique keys, no rows that are not label rows). The tests run it on every accepted and rewritten query, on 200 generated feature queries, and on the original of every query the guard had to rewrite, to show that it sees the leak.

## Columns that change after their row exists

A column is marked `mutable` in the schema graph (`Column.mutable`, `Column.mutable_source`) in three ways, and a feature query that reads a marked column is accepted with a warning naming it: `customers.is_churned may be overwritten after its row's event time (...)`. The warning is for the evidence report; nothing is rejected, because the guard cannot know whether the column really changes.

- **By name** (`mutable_name_reason`): `status`, `state`, `stage`, `is_*`, `has_*`, `*_flag`, `*_status`, `*_state`, `resolved*`, `closed*`, `final*`, `*_after_*`. Cheap and sometimes wrong in both directions: an `is_mobile` that never changes is flagged, a mutable column called `x1` is not.
- **By comparing two snapshots** (`ml/tasks/mutable_columns.py`, `POST /connections/{id}/schema/mutable-check`): for the rows that are in both snapshots (matched by primary key), a column whose value differs is mutable, whatever it is called. The result is saved in the schema overrides (`mutable_columns`). Only changes between the two snapshots are seen: a column that did not change in that time is not proven immutable, so the check is worth repeating as snapshots accumulate. Tables without a unique primary key are listed as skipped.
- **Confirmed by you** in the other direction: `immutable_columns` in the schema overrides (`PATCH /connections/{id}/schema`) clears the mark and the warning.

## Known limits

- The guard bounds each table by **its own event-time column**. A column that is filled in later than that time (a `resolved_at` or `delivered_at` next to an `ordered_at`, a status that is overwritten) still leaks through a correctly bounded query, and no SQL check can prove otherwise. The schema graph flags last-modified time columns (`time_leakage_hint`) and the guard refuses a table whose only time is a last-modified time (rule 8). For the late-filled columns of an otherwise well-timed table (such as `customers.is_churned`) the guard **accepts the query with a warning that names the column** (see "Columns that change after their row exists" below); the name-token and single-feature checks on the computed feature are a second layer, shown by the canaries.
- Static tables are the user's word. A table marked static whose rows grow or change (an `order_items` table whose rows are added with each order) leaks, and the runtime check does not see it because it does not truncate static tables. Only mark tables whose rows never change after the fact.
- A time-based or ordered pick among rows of equal time (`ORDER BY ordered_at DESC LIMIT 1` with ties) is not reproducible, though it can never come from after the cutoff.
- Event times are compared as UTC; a time-zone-aware column is converted by the label builder, but a feature query that compares it with the cutoff should convert it too.
- The runtime check needs the data in memory (a snapshot); on a live database only the SQL analysis runs.

## Canaries

`backend/tests/integration/test_canaries.py` plants seven traps in the demo database, shows that each is caught, and prints a "leakage canary report" (CI step "Leakage canaries"). The numbers are from a run on the demo database.

| # | Trap | Caught by |
|---|---|---|
| 1 | a feature from `customer_status_snapshot` (rewritten after churn, only time column `updated_at`), with or without a time bound | guard: `last_modified_time` (as written, the 'churned' flag has AUC 0.59) |
| 2 | a feature that is the label itself (orders in the 30 days after the cutoff) | guard: `reads_future`; if it got through, the single-feature check blocks it (score 0.99) |
| 3 | that table marked static to get past 1 | accepted, with the warning "static assumption on a table that changes" |
| 4 | `customers.is_churned` as a feature | name-token check (warn). The guard cannot see it, and the single-feature AUC is 0.59, below that check's limit |
| 5 | `MAX(ordered_at)` without a cutoff filter | guard rewrite; the original differs when the future is deleted in 563 sampled rows, the rewritten one in 0 |
| 6 | a random split on the relational task | refused (#52) |
| 7 | `customers.discount_code_used_after_churn` as a feature | name-token check (warn); single-feature AUC 0.56 |

Limit shown by 4 and 7: the guard bounds rows, not the columns of a row. A column that is filled in after the row's own event time, in a table that is otherwise well timed, passes the SQL guard; only the name check sees it, so a column with an innocent name would pass. Checking mutable columns is not built yet. The name check also assumes the training table's label column carries the task's name (here `churn_30d`); the training-table builder must do that.
