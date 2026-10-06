# 0002. DuckDB as the one internal engine

**Status:** Accepted. Built for files in #42: uploads, samples and SQL-query snapshots become tables in a per-project DuckDB file and everything reads them through one `DataSource` interface. `planned (#43, #97)`: snapshots of whole live databases.

## Context

Every data source has to end up somewhere the training-table builder, the feature queries and the point-in-time guard can all work on with one SQL dialect. Running those on a customer's production database is slow, risky and different on every database.

## Decision

Read from the user's database through the read-only connection (0003), snapshot what a run needs into a local DuckDB file, and run labels, features and checks there. CSV and Parquet files are treated as a one-table database. The snapshot is the data version (#97), so a run says exactly what it trained on.

## What #42 settled

- One file per project, `data/projects/<id>/work.duckdb`, one table per data version named after the file. The read-only content-hashed copies in `data/versions/` stay the record of what was loaded; the table is the working copy.
- The project file is opened with `enable_external_access=false`, so a query that reaches it cannot read or write other files, attach databases or load extensions. Files are read on a separate in-memory connection and streamed in as Arrow batches. `DataSource.query` also accepts a single `SELECT` only, with a row limit and a timeout.
- CSV columns are typed like pandas would type them (integers, floats, text; the same list of null spellings). DuckDB's own inference is not used for booleans and dates: it would turn a `Yes`/`No` target into booleans and change the class labels. Dates stay text until a task spec names the event-time column.
- Results are Arrow. They become pandas only where a model needs a frame.

## Consequences

- One dialect for everything we generate and guard.
- Fast analytic queries on a laptop, with no load on the source after the snapshot.
- Large databases need sampling or windowed snapshots; that cost is real and is designed into #97.
- Data is copied onto the machine running MLPilot. That fits self-hosting (0008), but the data-handling note for pilots must say it.

## Alternatives considered

- **Query the source database directly.** Dialect differences and load on production. Rejected as the default.
- **pandas only.** No SQL to show the user, no way to guard queries. Rejected.
- **SQLite as the engine.** Weaker for analytics and window functions. Rejected.

## Issues

#42 (engine), #97 (snapshots and versions), #43 (connections).
