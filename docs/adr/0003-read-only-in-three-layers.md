# 0003. Read-only access to user databases, enforced in three layers

**Status:** Accepted and implemented in `backend/ml/data/sql_guard.py` (#44), the only module that executes SQL on a user database (saved connections and the old connection-string import alike). Showing the `can_write` warning in the UI is `planned (#46, #103)`.

## Context

MLPilot is connected to a company's data. One bug, one clever prompt injection or one wrong credential must not be able to change that data.

## Decision

Every query against a user database passes three independent checks, so any one of them can fail without harm:

1. **Parser allowlist.** `sqlglot` parses the text; it must be a single `SELECT` (or `UNION`/`INTERSECT`/`EXCEPT`, with CTEs). Inserts, updates, deletes, merges, DDL, grants, `SELECT ... INTO`, `COPY`, `SET`, transaction control, locking clauses (`FOR UPDATE`) and commands (`CALL`, `DO`, `EXPLAIN`, ...) are refused anywhere in the tree, including inside a CTE. Side-effect functions (`pg_sleep`, `set_config`, `nextval`, `lo_import`, `dblink*`, `pg_read_file`, `query_to_xml`, DuckDB's `read_*`, SQLite's `load_extension`, ...) and system tables holding secrets (`pg_authid`, `pg_shadow`, `pg_stat_activity`, ...) are refused. A snapshot through a saved connection may read only tables the connection's schema lists. Each refusal has a stable code. The database runs the SQL re-rendered from the checked tree, without comments, so it runs what was checked.
2. **Read-only session with limits.** Postgres: every statement runs in `START TRANSACTION READ ONLY` with `statement_timeout` and `idle_in_transaction_session_timeout`, always rolled back, and over the extended protocol, which accepts one statement per call (so a smuggled `ROLLBACK; DELETE` is refused by the server). SQLite: the file is opened `mode=ro`, with `query_only`, no attached databases, an authorizer that allows reads only, and a deadline. DuckDB: the file is opened `read_only`, without access to other files, with its configuration locked, one statement per call, and an interrupt timer. Every query is wrapped in a row limit (more rows raise, never truncate) and a result-size limit. Each executed query is logged as the SHA-256 of its text, its duration, row count and size; never its text or values.
3. **Read-only credentials.** The connection test checks whether the role could write anything (superuser, create rights, write grants on any table) and saves `can_write`, with the reasons. The pilot kit gives customers the SQL to create a read-only role (#103).

Passwords and connection strings are never logged or returned; errors are redacted.

## Consequences

- Some valid read-only SQL is refused until the parser allowlist learns it. Safe by default over convenient.
- Layers 2 and 3 depend on each database's features, so each supported database needs its own test.
- Layer 2 does not stop a side effect that is not a write, such as `COPY ... TO PROGRAM` run by a superuser; only layer 1 and a non-superuser role (layer 3 reports superusers) do. That is why the role check matters.
- Only Postgres, SQLite and DuckDB are supported; the old connection-string import no longer reaches other databases through SQLAlchemy.
- Writing predictions back to a user database is a separate, opt-in feature on a separate connection (#71), never part of this path.

## Alternatives considered

- **Trust the credentials alone.** One misconfigured role defeats it. Rejected.
- **Regex or keyword filtering.** Easy to bypass. Rejected for parsing.

## Evidence and issues

`tests/fixtures/hostile_sql.yaml` (the hostile-query matrix with expected codes), `test_sql_guard.py`, `test_sql_guard_postgres.py` (the matrix against Postgres in CI; writes refused with layer 1 switched off on Postgres, SQLite and DuckDB; `can_write` for a role with INSERT), `test_sql_loader.py`, `test_api_security.py` (passwords do not leak). Issues: #30, #43, #44, #103.
