# Export bundle (#61)

`GET /api/v1/projects/{id}/runs/{run}/export[?dialect=postgres|duckdb]` returns a zip for a run that ended (`completed` or `stopped`) and kept its model. The default dialect is the source database's if it is Postgres or DuckDB, otherwise DuckDB (the bundle's README says so).

```
mlpilot_export_<run>/
  README.md                  how to run, what each file is, the honest metrics
  task.yaml                  the confirmed spec
  manifest.json             the run's manifest + the pinned library versions
  report.html                the evidence report (#60), offline
  entities.sql               the entities eligible at one cutoff (__CUTOFF__ is filled in by score.py)
  features/<name>.sql        one query per champion feature; reads __labels(entity_id, cutoff_time)
  dbt/                       dbt project: models/entities.sql, models/features/f_<name>.sql, models/features_all.sql
  model/model.txt            LightGBM native text format
  reference_validation.csv   the validation rows with the scores MLPilot computed
  bundle.json                dialect, feature order, text-column levels, the reported validation metrics
  score.py, sqlcheck.py      the scoring script and the guard rules it applies
  requirements.txt           pinned
```

## How it is built

- The run service saves `model.txt` (the champion fitted on the training rows, stopped on the validation rows), `reference_validation.csv` and `categories.json` (the levels of text columns, which LightGBM needs again) under `projects/<project>/runs/<run>/` when a run ends. With rollouts, those of the chosen rollout.
- Feature SQL is the stored DuckDB text, transpiled with sqlglot when the target is Postgres.
- `entities.sql` is the `eligible` step of the task's own label query with the cutoffs replaced by one placeholder, so what is scored is what was trained on.
- `score.py --cutoff D` runs `entities.sql`, then each feature query with `__labels` prepended as a CTE, then the model. `--verify` does the same for the rows of `reference_validation.csv` and compares the PR-AUC with the one MLPilot reported (tolerance 1e-6).
- `score.py` reads `MLPILOT_DB_URL` (`postgresql://user:password@host:port/db` or `duckdb:///path.db`), opens the connection read-only (Postgres: `SET TRANSACTION READ ONLY`; DuckDB: `read_only=True`), and passes every statement through `sqlcheck.py` first. The URL is never printed. `tests/unit/test_export_sqlcheck.py` runs the main guard's hostile matrix against `sqlcheck.py` and fails if its lists differ from `ml/data/sql_guard.py`.

## What the tests show

- `test_export_bundle.py`: an exported scripted run reproduces the validation PR-AUC from a DuckDB copy of its data (difference in a single score about 1e-16 over 1559 rows) and scores every eligible entity at a new cutoff; the same on the demo Postgres, as the read-only role, through pg8000 (CI job `demo-db`).
- CI job `export`: the unzipped bundle is installed in a clean virtualenv from its own `requirements.txt` and `score.py --verify` is run; `dbt compile` runs on its dbt project with `dbt-duckdb`.

## Limits

- The model in the bundle is the one the validation numbers belong to. The test score of the run comes from a refit on train and validation, which is not exported.
- The queries use the table names of the schema graph, with no schema prefix; put the schema on the connection's `search_path`.
- The dbt models compile; running them against a warehouse other than DuckDB is not tested.
- SQLite sources are exported for DuckDB: load the tables into DuckDB first.
- A run made before this change kept no model file and answers 409.
