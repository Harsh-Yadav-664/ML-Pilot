# 0003. Read-only access to user databases, enforced in three layers

**Status:** Accepted. Layers 1 and 2 are implemented for the SQL import; the full connection manager and privilege check are `planned (#43, #44)`.

## Context

MLPilot is connected to a company's data. One bug, one clever prompt injection or one wrong credential must not be able to change that data.

## Decision

Every query against a user database passes three independent checks, so any one of them can fail without harm:

1. **Parser allowlist.** `sqlglot` parses the text; it must be a single `SELECT`. Inserts, updates, deletes, DDL, `SELECT ... INTO` and commands are refused anywhere in the tree, including inside a CTE.
2. **Read-only transaction with limits.** The query is wrapped in a row limit on every database and runs in a read-only transaction (`SET TRANSACTION READ ONLY` on Postgres and MySQL, `PRAGMA query_only` on SQLite) with a statement timeout on Postgres and MySQL. A timeout for SQLite is not set yet.
3. **Read-only credentials.** The connection manager checks that the role has no write privileges and warns if it does (`planned (#44)`). The pilot kit gives customers the SQL to create such a role (#103).

Passwords and connection strings are never logged or returned; errors are redacted.

## Consequences

- Some valid read-only SQL is refused until the parser allowlist learns it. Safe by default over convenient.
- Layers 2 and 3 depend on each database's features, so each supported database needs its own test.
- Writing predictions back to a user database is a separate, opt-in feature on a separate connection (#71), never part of this path.

## Alternatives considered

- **Trust the credentials alone.** One misconfigured role defeats it. Rejected.
- **Regex or keyword filtering.** Easy to bypass. Rejected for parsing.

## Evidence and issues

`test_sql_loader.py`, `test_api_security.py` (passwords do not leak). Issues: #30, #43, #44, #103.
