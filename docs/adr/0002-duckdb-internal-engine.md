# 0002. DuckDB as the one internal engine

**Status:** Accepted. `planned (#42)`: nothing in the code uses DuckDB yet; today data is a CSV loaded with pandas.

## Context

Every data source has to end up somewhere the training-table builder, the feature queries and the point-in-time guard can all work on with one SQL dialect. Running those on a customer's production database is slow, risky and different on every database.

## Decision

Read from the user's database through the read-only connection (0003), snapshot what a run needs into a local DuckDB file, and run labels, features and checks there. CSV and Parquet files are treated as a one-table database. The snapshot is the data version (#97), so a run says exactly what it trained on.

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
