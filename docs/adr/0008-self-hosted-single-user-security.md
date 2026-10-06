# 0008. Self-hosted, single-user security model

**Status:** Accepted. Implemented (#92).

## Context

MLPilot holds database connections. A tool that is open to any website in the user's browser, or to anyone on the same network, is not safe to point at company data. Building user accounts and roles is a large project nobody has asked for.

## Decision

- Listen on `127.0.0.1` by default. Any other host needs `MLPILOT_HOST`, and starting that way without an explicit token logs a warning.
- Every API route needs a local access token (`Authorization: Bearer`). It comes from `MLPILOT_TOKEN` or a random token kept in `~/.mlpilot/token` with owner-only permissions. `python start.py` passes it to the UI, so local use needs no setup.
- CORS allows only the configured UI origins.
- This is not multi-user authentication: there are no accounts, roles or per-user data.

## Consequences

- A website open in the same browser cannot call the API, and another machine cannot reach it unless the owner opted in.
- The token is readable by the UI's JavaScript; it protects against other sites and hosts, not against code running as the same user.
- If teams want to share one deployment, that needs a new ADR (accounts, roles, an audit log).

## Alternatives considered

- **No authentication, localhost only.** Does not stop other browser tabs. Rejected.
- **Full accounts and SSO.** Out of scope until someone needs them.

## Evidence and issues

`test_local_token_and_cors.py`, CI `ui-smoke` (a fresh `python start.py`). Issue: #92.
