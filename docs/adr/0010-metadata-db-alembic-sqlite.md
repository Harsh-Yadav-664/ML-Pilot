# 0010. Metadata database owned by Alembic, SQLite by default

**Status:** Accepted. Implemented (#89, #91).

## Context

MLPilot stores its own state: projects, data versions, runs, experiments, features, jobs. Users upgrade between versions, so the schema must change without losing rows or making users run SQL.

## Decision

- Alembic owns the schema. Tables are never created from the models directly. The app runs `alembic upgrade head` on start-up, and a database made before migrations existed is stamped at the first revision so its rows are kept.
- SQLite is the default (one file, no setup); Postgres works through `DATABASE_URL` for shared deployments.
- A test fails if the models and the migrations disagree.
- Anything that must survive a restart (jobs, events, manifests) lives in this database, not in memory.

## Consequences

- Every model change needs a reviewed migration.
- SQLite limits concurrent writes; the job runner and tests keep write transactions short. Postgres is the answer if that becomes a limit.

## Alternatives considered

- **`create_all` at start-up.** No upgrade path. Rejected.
- **Postgres only.** Forces every user to run a database before they can try the tool. Rejected.

## Evidence and issues

`test_migrations.py`, `test_job_runner.py`. Issues: #89, #91.
