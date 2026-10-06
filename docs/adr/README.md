# Architecture decision records

One short page per decision that someone would otherwise re-open: what was decided, why, what it costs, and what was rejected. Format: Status, Context, Decision, Consequences, Alternatives considered, and the issues that implement it.

**Status** says what is true today. "Implemented" means a test covers it; anything else is labelled `planned (#N)`. If you change a decision, add a new ADR that supersedes the old one instead of editing history.

| ADR | Decision |
|---|---|
| [0001](0001-llm-proposes-code-decides.md) | The LLM proposes; deterministic code validates, executes and decides |
| [0002](0002-duckdb-internal-engine.md) | DuckDB as the one internal engine |
| [0003](0003-read-only-in-three-layers.md) | Read-only access to user databases, enforced in three layers |
| [0004](0004-point-in-time-guard-by-sql-analysis.md) | The point-in-time guard analyses the SQL, instead of trusting prompt rules |
| [0005](0005-feature-ir-first-free-sql-second.md) | A typed feature spec first, free SQL second |
| [0006](0006-temporal-validation-and-paired-acceptance.md) | Time-based validation and a paired acceptance rule |
| [0007](0007-lightgbm-default-licence-policy.md) | LightGBM by default, optional engines, permissive licences only |
| [0008](0008-self-hosted-single-user-security.md) | Self-hosted, single-user security model |
| [0009](0009-privacy-by-default.md) | The LLM sees schema and aggregates, not rows |
| [0010](0010-metadata-db-alembic-sqlite.md) | Metadata database owned by Alembic, SQLite by default |
