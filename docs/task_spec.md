# Task spec reference

A task spec says what to predict, for whom, and from which dates. It is a small YAML document. MLPilot checks it against the schema of your database before anything is trained, and every label in the training table is built from it (the label builder is issue #50).

Unknown keys are errors: a typo is reported, never ignored.

```yaml
name: churn_30d
description: Customers who were active in the last 90 days and place no order in the next 30.
entity: {table: customers, key: customer_id, created_at: signup_at}
eligibility:
  - "signup_at < :cutoff"
  - exists: {table: orders, where: "ordered_at >= :cutoff - interval '90 days' AND ordered_at < :cutoff"}
target:
  type: binary
  window: {start: ":cutoff", end: ":cutoff + interval '30 days'"}
  expression: {table: orders, agg: count, where: "status != 'cancelled'", compare: "= 0"}
horizon: 30d
cutoffs: {start: 2023-04-01, end: 2024-10-01, every: 1 month}
split: {val_from: 2024-04-01, test_from: 2024-07-01}
metric: pr_auc
```

## Fields

| Field | Meaning |
|---|---|
| `name` | Lower-case letters, digits and underscores. A task keeps its name across versions. |
| `entity` | Who the prediction is about. `table` and `key` are required; `created_at` (optional) is when the entity came to exist. |
| `eligibility` | Who is scored at a cutoff. A list; every entry must hold. An entry is a condition on the entity table (see below) or `exists:` / `not_exists:` with `table`, optional `where` and optional `via` (the column of that table that holds the entity key, needed only if it has several). |
| `target.type` | `binary` (trainable now), `regression` or `multiclass` (accepted and flagged: the trainer refuses them until #104). |
| `target.expression` | `table`, `agg` (`count`, `sum`, `avg`, `min`, `max`, `count_distinct`), `column` (all but `count`), optional `where`, and for binary tasks `compare` (the test that makes the label 1, such as `= 0` or `>= 3`). The aggregate is taken over that table's rows of the entity in the label window. |
| `target.window` | Optional. `start` is `:cutoff` (a label never looks before the cutoff). `end` is `:cutoff + interval 'N days'` (hours, days or weeks) and may not be later than the horizon. Default: the window is `(cutoff, cutoff + horizon]`. |
| `target.expression_sql` | Escape hatch instead of `expression`: one `SELECT` that uses `:cutoff`. Accepted with a warning. It runs through the SQL guard and the point-in-time guard, is not understood by the label checks, and is flagged in the evidence report. |
| `horizon` | How far ahead the label looks: `30d`, `4 weeks`, `1 month`. |
| `cutoffs` | `start`, `end` and `every` (`7d`, `2 weeks`, `1 month`). Cutoff dates are `start`, `start + every`, ... up to `end`. Months keep the day of the month (clamped to short months). |
| `split` | `val_from` and `test_from`. Training uses cutoffs whose label window ends before `val_from`; validation those from `val_from` whose window ends before `test_from`; test those from `test_from`. |
| `metric` | `binary`: `pr_auc` (default), `roc_auc`, `f1`, `log_loss`. `regression`: `mae` (default), `rmse`, `r2`. `multiclass`: `macro_f1` (default), `accuracy`, `log_loss`. |

## Conditions

A condition is a `WHERE`-style test on the columns of one table: `status != 'cancelled' AND total > 10`. It may use comparison operators, `AND`, `OR`, `NOT`, `IN`, `IS NULL`, `LIKE`, `BETWEEN`, literals, and the placeholder `:cutoff` with an interval (`:cutoff - interval '90 days'`). Functions, casts, subqueries, other tables, other placeholders and more than one statement are refused. Conditions are parsed with sqlglot and checked against a whitelist, so what the label builder compiles is exactly what was checked.

## What is checked

Against the schema graph of the connection (`POST /projects/{id}/tasks/validate` returns every problem at once, each with the path of its field):

- tables and columns exist; the entity key is the table's primary key (warning if not); `created_at` is a date or timestamp;
- every table the spec reads (`exists`, `not_exists`, `target.expression`) has a direct foreign key to the entity table, and has an event-time column so its rows can be limited to before the cutoff or to the label window;
- `sum` and `avg` use a numeric column; binary targets have a `compare`; others do not;
- the window ends no later than the horizon and does not start before the cutoff;
- the last cutoff plus the horizon lies before the end of the data (the `as_of` of the database version, or now), so no label is incomplete;
- `val_from` and `test_from` lie inside the cutoff range, are at least one horizon apart, and leave at least one cutoff for training, validation and test.

Tables more than one foreign key away from the entity (for example `order_items` through `orders`) are not supported yet; the error says so.

## Versions

`POST /projects/{id}/tasks/` saves a draft (version 1). A draft is edited in place with `PUT`. `POST .../confirm` validates again and refuses if any error remains. A confirmed spec is never changed: editing it with `PUT` saves the next version as a new draft, and every run that points at the confirmed version keeps pointing at it. The confirmed row stores the schema fingerprint it was checked against.
