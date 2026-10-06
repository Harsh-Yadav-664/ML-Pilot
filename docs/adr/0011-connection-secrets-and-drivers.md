# 0011. Saved connections: secrets stay out of the database and the logs, and the Postgres driver is pg8000

**Status:** Accepted and implemented in #43 (Postgres, SQLite and DuckDB files).

## Context

A saved connection needs a password, and a password must not end up in the metadata database, an API response or a log line (AGENTS.md rule 7). The Postgres driver also has to be allowed by the licence rule (rule 9). The usual choice, psycopg 3, is LGPL-3.0, which is not on the permissive list.

## Decision

- **Where the password lives.** The `connections` row holds a `secret_ref`, never the password. Either `env:NAME` (the password is read from that environment variable when needed, so MLPilot stores nothing secret) or `enc:<token>` (the password encrypted with Fernet under the key in `MLPILOT_SECRET_KEY`). Without that key MLPilot refuses to store a password and says so. An `env:` reference can't name MLPilot's own secrets (`MLPILOT_*`, `*_API_KEY`, `*_TOKEN`), so a saved connection can't be used to send them to a database host.
- **What the API says.** A response carries `secret: "env:NAME"` or `"stored (encrypted)"`, never a value. The test endpoint reports failures as `ok: false` with a short code (`auth_failed`, `host_unreachable`, `timeout`, `ssl_required`, `ssl_failed`, `database_not_found`) and a fixed message. A driver's own error text is dropped because it can contain host names or parts of the connection string.
- **Logs.** Every secret MLPilot reads or stores is registered with a log-record factory that masks it, and any `scheme://user:password@host` URL, in every record and traceback from every logger. A test runs a full connect, test and query cycle at DEBUG level and scans the captured logs, output and responses for the password.
- **Driver.** `pg8000` (BSD-3, pure Python) for Postgres. SQLite through the standard library with `mode=ro` and `PRAGMA query_only`; DuckDB with `read_only=True` and no access to other files. `psycopg` 3 is faster, but only switch if a benchmark shows pg8000 is the bottleneck, and then record it in a new ADR with a way to keep the licence rule intact (an optional install, not a default).
- **Role privileges.** The test endpoint reports whether the role could write (superuser, create rights, write grants) and saves it as `can_write`. This is the input to layer 3 of [ADR 0003](0003-read-only-in-three-layers.md); acting on it (warn or block) is #44.

## Consequences

- A user who wants stored passwords has to create a Fernet key once. Not having one is not a blocker: `password_env` works without it.
- Changing `MLPILOT_SECRET_KEY` makes stored passwords unreadable. The error says so and names the key, not the password.
- Secrets under 4 characters are not masked in logs, because masking them would garble every line that contains those letters.
- pg8000 is slower than psycopg on large result sets. Snapshots are capped by a row limit, so this has not mattered yet.
- Foreign keys of live Postgres and SQLite tables are not read yet; the schema graph is #45.

## Alternatives considered

- **Store the password in plain text in the metadata DB.** Rejected: rule 7.
- **Use the operating system keychain.** Rejected for v1: it differs per platform and does not work in a container.
- **psycopg 3 by default.** Rejected for the licence, not for quality.
- **A log filter on handlers.** Rejected for a record factory: a filter misses handlers added later, such as a test's capture handler.

## Evidence and issues

`test_secrets_and_redaction.py`, `test_live_sources.py` (read-only at the database level, privilege check, error mapping), `test_connections_api.py` (create, test, list, delete, DEBUG log scan, refusal without a key). Issues: #43, #44, #45.
