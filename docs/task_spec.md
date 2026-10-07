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
| `split` | `val_from` and `test_from`. Training uses cutoffs before `val_from` whose label window ends by `val_from`; validation those from `val_from` whose window ends by `test_from`; test those from `test_from`. Cutoffs whose window crosses the next boundary are left out and counted (see [How the split works](#how-the-split-works)). |
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

## How the split works

A relational task is split by cutoff date, never at random (`ml/validation/splits.py`, `make_temporal_splits`). A random split would put rows of the same customer at nearby cutoffs on both sides, and a training label could look into the test period.

- **train**: cutoff before `val_from` and label window end at or before `val_from`. Its label sees nothing of the validation period.
- **validation**: cutoff in `[val_from, test_from)` and window end at or before `test_from`.
- **test**: cutoff at or after `test_from`. It is scored once, at the end of a run.
- A cutoff whose window crosses the next boundary is *purged*: it is in neither part, and the counts are reported (`n_purged_train`, `n_purged_val`). A window that ends exactly on the boundary is kept; one second later is purged.
- Model selection inside training uses expanding folds over the training cutoffs: each fold trains on earlier cutoffs only and validates on later ones, and every training window in a fold ends by the start of that fold's validation cutoffs.
- Asking for a `holdout`, `cv` or any other non-temporal strategy through `split_plan_for_task`, or `make_splits(..., relational=True)`, raises `SplitError` with the reason; there is no way to opt in. The refusal lives in those two functions: the training pipeline for relational tasks does not exist yet (#55 onward) and must call them, and the older single-table experiment path keeps its random splits.
- The same entity appears on both sides of a boundary (a customer at several cutoffs). That is normal for forecasting; when entity ids are passed, the split summary counts the entities shared by train and validation, train and test, and validation and test.
- A missing (`NaT`) or unreadable cutoff or window end raises `SplitError`; times with a UTC offset are converted to UTC and times without one are read as UTC.

`POST /projects/{id}/tasks/{task_id}/preview-split` shows the result before a run: a timeline of cutoffs with their part and row count, the totals, the latest training and validation window end, and the folds. It uses the same functions as training, on the label counts of `preview-labels`.

## Feasibility checks

Before a run starts, the label table is checked (`ml/tasks/feasibility.py`). Each check returns `ok`, `warn` or `block` with the numbers it used; the report is part of `preview-labels` (`feasibility`), and is stored in the run's manifest. A value exactly at a limit passes.

| Check | Warns | Blocks |
|---|---|---|
| `positives_per_split` (binary) | any of train, validation, test has fewer than 200 positives | train has fewer than 50 positives, or validation or test fewer than 20 |
| `base_rate_per_cutoff` (binary) | a cutoff has no positives, or the highest base rate is more than 3 times the lowest | |
| `eligible_per_cutoff` | eligible entities fall by more than 50% between adjacent cutoffs | |
| `class_imbalance` (binary) | the rarer class is under 1% of rows; the report suggests PR-AUC (lift at k is not available yet, #93) | |
| `coverage` | at the first cutoff, no related table has earlier rows for 5% of the eligible entities, or there is no related table at all | |
| `horizon_vs_data` | cutoffs were dropped because their label window ends after the data does | |

A related table is one with an event time and a direct foreign key to the entity table. Coverage counts entities with at least one row *before* the first cutoff, so it reads no future data. On a snapshot it covers only the tables and columns the snapshot holds. The limits are a `thresholds` object in the request (`min_train_positives`, `min_eval_positives`, `warn_positives`, `max_base_rate_ratio`, `max_eligible_drop`, `imbalance_rate`, `min_coverage`); the ones used are in the report.

`POST /projects/{id}/tasks/{task_id}/runs` records a run of a confirmed task. If a check blocks, the answer is a 422 with the reasons and the report, unless the body has `override: true`; then the run is recorded and `manifest.override` keeps the reason you gave, what blocked and when. This records the run only: training on a task is built in #55 onward.

## Versions

`POST /projects/{id}/tasks/` saves a draft (version 1). A draft is edited in place with `PUT`. `POST .../confirm` validates again and refuses if any error remains. A confirmed spec is never changed: editing it with `PUT` saves the next version as a new draft, and every run that points at the confirmed version keeps pointing at it. The confirmed row stores the schema fingerprint it was checked against.

## Labels at the cutoff dates

`POST /projects/{id}/tasks/{task_id}/preview-labels` builds the labels of a saved task (`ml/tasks/labels.py`). The body is optional: `data_version_id` picks a database snapshot (the labels are then also written to the project's DuckDB file as `mlpilot.labels_<task>_<data version>`; `"materialize": false` skips that) or a live version; without it the labels are computed read-only on the connected database through the SQL guard. SQLite sources need a snapshot. The answer holds the generated SQL, per cutoff the eligible entities, positives and base rate (mean label for regression), the cutoffs left out, and the table written.

One row per (entity, cutoff): `entity_id, cutoff_time, label, label_window_end`.

- **Eligible** means the entity existed before the cutoff (`created_at < cutoff`), every eligibility condition holds, and every `exists` / `not_exists` test holds over that table's rows **strictly before** the cutoff. The compiler adds that time bound itself, so an eligibility rule cannot read the future even if the spec forgot to limit it.
- **The label window is `(cutoff, window_end]`.** An event exactly at the cutoff is not in the window (it is history for that cutoff); an event exactly at the window end is. Feature history uses strictly before the cutoff. This is the convention the tests pin down, so a feature and a label never share an event.
- Entities with no event in the window still get a row: a count and a sum are 0 for them.
- Rows whose label is NULL (an `avg`, `min` or `max` over no events) are left out, not turned into 0.
- A cutoff whose window ends after the data does is **dropped and reported** (`dropped_cutoffs`), never kept with an incomplete label. Saving or confirming a spec still treats that as an error; the preview only drops them so a spec written for newer data can be tried on an older snapshot.
- Times are compared as UTC. A time-zone-aware column is converted; a column without a zone is taken to hold UTC; text times (SQLite) are parsed.
- `target.expression_sql` is not compiled here: it needs the point-in-time guard (#51).
