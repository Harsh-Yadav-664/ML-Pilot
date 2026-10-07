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

`assumptions` lists static tables that were read without a time bound. They appear in the evidence report.

## The rules

1. The query is one read-only `SELECT` (it goes through the SQL guard first), is parsed with sqlglot and every column is qualified, so each reference names its table.
2. **Every table with an event time needs a bound in its own scope.** A top-level `AND` term of that scope's `WHERE`, or of the table's own `JOIN ... ON`, of the form `t.time < K`, or `t.time <= K - INTERVAL 'n ...'` with the interval at least one microsecond (a smaller one rounds to zero in the database). `K` is the cutoff. Bounds inside an `OR`, bounds on another table, `<=` the cutoff itself, comparisons on a `DATE` cast, casts with a precision such as `TIMESTAMP(0)` and a column compared with itself do not count. A date-typed event column has no time of day, so it is compared in whole days: `t.d < CAST(K AS DATE)`.
3. **A table without an event time** is accepted only if the user confirmed it as static in the schema. Otherwise the guard says so and names the table.
4. **Which cutoff.** `K` must be `__labels.cutoff_time` or a pass-through of it. A scope's cutoff sources are its own `__labels`, derived tables and CTEs that expose a cutoff column, and, for a correlated subquery, the enclosing row. They must be joined to each other on `cutoff_time` equality (in the `WHERE`, or in the `ON` of one of the two joins involved). Otherwise a derived table could hand over values that were computed for another cutoff.
5. **No mixing of label rows.** In a scope that reads its own cutoff, aggregates must `GROUP BY` the cutoff, `DISTINCT` must keep it in the output, window functions must `PARTITION BY` it, and `LIMIT` is refused (a `LIMIT` in a correlated subquery acts per outer row and is fine). A derived table or CTE that reads `__labels` must expose `cutoff_time`.
6. **Refused outright:** unknown tables and label tables; non-deterministic functions (`now()`, `current_date`, `random()`, `uuid()`, literals such as `'now'`) and any function the SQL parser does not know by name (it could be one of them); order-dependent picks (`any_value`, `first`, `last`, `array_agg` and `string_agg` without `ORDER BY`, `LIMIT` without `ORDER BY`); window frames with `FOLLOWING`; `UNION`/`INTERSECT`/`EXCEPT`; lateral joins, `UNNEST`; every join except `JOIN`, `INNER JOIN`, `CROSS JOIN` and `LEFT JOIN` with `ON` (so no `SEMI`, `ANTI`, `ASOF`, `POSITIONAL`, `NATURAL`, `USING`, right or full joins); `GROUPING SETS`/`ROLLUP`/`CUBE`; `DISTINCT ON`; a second `__labels` in one scope; two output columns with one name; aliases that differ only in case.
7. **Rewriting.** When a bound is missing and the scope has a cutoff to use, the guard adds `t.time < <cutoff>`: to the table's own `ON` when it is joined, otherwise to the `WHERE` (never when an outer join would then drop label rows). The cutoff it names must resolve, from that scope, to the cutoff it picked (a name shadowed by an inner alias is refused), and inside an `ON` it must belong to a source joined earlier. Anything else that is wrong is rejected, not repaired.

Everything the guard does not model is rejected. The test fixture `backend/tests/fixtures/pit_queries.yaml` lists 70 queries with their expected result.

## The runtime check

`pit_verify.truncation_check` recomputes a feature on a copy of the data in which every row at or after the cutoff has been deleted, for a sample of label rows at several cutoffs, and compares the values. A difference is a leak the SQL analysis missed and is reported with the entity, the cutoff and both values. It also enforces the contract (columns, unique keys, no rows that are not label rows). The tests run it on every accepted and rewritten query, on 200 generated feature queries, and on the original of every query the guard had to rewrite, to show that it sees the leak.

## Known limits

- The guard bounds each table by **its own event-time column**. A column that is filled in later than that time (a `resolved_at` or `delivered_at` next to an `ordered_at`, a status that is overwritten) still leaks through a correctly bounded query. The schema graph flags last-modified time columns (`time_leakage_hint`); a column-level check is #54.
- Static tables are the user's word. A table marked static whose rows grow or change (an `order_items` table whose rows are added with each order) leaks, and the runtime check does not see it because it does not truncate static tables. Only mark tables whose rows never change after the fact.
- A time-based or ordered pick among rows of equal time (`ORDER BY ordered_at DESC LIMIT 1` with ties) is not reproducible, though it can never come from after the cutoff.
- Event times are compared as UTC; a time-zone-aware column is converted by the label builder, but a feature query that compares it with the cutoff should convert it too.
- The runtime check needs the data in memory (a snapshot); on a live database only the SQL analysis runs.
