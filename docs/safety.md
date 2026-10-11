# Safety model

What MLPilot guarantees, and the test that shows each one. If a line here has no test next to it, it is not a guarantee.

| Guarantee | How | Test or CI step |
|---|---|---|
| It never writes to your database | Four layers, any one of which holds alone: your role's grants, a SQL parser that accepts one `SELECT` only, a read-only session with a timeout, and a row limit that raises instead of truncating. The connection test warns when the role could write. | [`test_sql_guard.py`](../backend/tests/unit/test_sql_guard.py) (40+ hostile queries), [`test_sql_guard_postgres.py`](../backend/tests/integration/test_sql_guard_postgres.py), [`test_pilot_role.py`](../backend/tests/integration/test_pilot_role.py); layer table in [data handling](pilot/data_handling.md#4-the-read-only-layers) |
| A training row never uses data from after its cutoff | Every feature's SQL is read by the point-in-time guard, which refuses a table with an event time that is not bounded by the cutoff. A second check recomputes the result independently. | [`test_pit_guard.py`](../backend/tests/unit/test_pit_guard.py), [`test_llm_sql.py`](../backend/tests/unit/test_llm_sql.py) (a leaky proposal is refused); [contract](pit_guard.md) |
| Relational tasks are never split at random | Splits are by cutoff date; windows that cross a boundary are dropped and counted. | [`test_temporal_split.py`](../backend/tests/unit/test_temporal_split.py); [rules](task_spec.md#how-the-split-works) |
| The test set is scored once and never used to choose | Model and feature selection use earlier periods only; the test rows are predicted once, at the end. | [`test_honest_split.py`](../backend/tests/unit/test_honest_split.py) |
| A feature is kept by code, not by the model | The model proposes; a fixed rule on validated gains accepts or rejects. | [`test_acceptance.py`](../backend/tests/unit/test_acceptance.py) |
| Model output is never executed | Formulas go through a whitelist evaluator; SQL goes through the guards above. | [`test_safe_eval.py`](../backend/tests/unit/test_safe_eval.py) |
| The model sees no cell values by default | Every prompt comes from one builder with four privacy levels; the default shows names and aggregates. Every prompt is logged. | [`test_context_builder.py`](../backend/tests/unit/test_context_builder.py), [`test_prompt_privacy_demo_db.py`](../backend/tests/unit/test_prompt_privacy_demo_db.py); [levels](pilot/data_handling.md#3-what-a-language-model-is-shown) |
| Passwords and keys do not leak | The password is read from an environment variable, never logged or returned. | [`test_cli.py`](../backend/tests/integration/test_cli.py), [`test_api_security.py`](../backend/tests/integration/test_api_security.py) |
| A failure is reported, not hidden | A failed run records the reason; a fallback to the offline stub is labelled `fallback`. | [`test_ai_gateway.py`](../backend/tests/unit/test_ai_gateway.py), [`test_job_runner.py`](../backend/tests/integration/test_job_runner.py) |

## What is not covered

- Network access to your database (firewall, TLS settings): configure these on your side.
- What a hosted language-model provider does with a prompt. With no provider key set no model is contacted, and a local model keeps prompts on your machine.
- What you type (the question, chat messages) is sent as typed, apart from columns marked "never send".
- Measured accuracy of the drafted task on real models: see [`scripts/nl_accuracy.py`](../backend/scripts/nl_accuracy.py); no number is claimed until it has run.
