# MLPilot pilot: what happens to your data

MLPilot runs on a machine you choose. This page says what it reads from your database, what it keeps, what a language model is shown, and what is logged. **Every statement ends with the test or code that shows it.** "Not tested" means no test shows it, and the sentence is only what the code is meant to do. Tests are in [`backend/tests/`](../../backend/tests/); the ones marked CI run in the [CI workflow](../../.github/workflows/ci.yml). Reasoning behind the design: [ADR 0003](../adr/0003-read-only-in-three-layers.md) (read-only), [ADR 0009](../adr/0009-privacy-by-default.md) (what the model sees).

## 1. What MLPilot reads

| Statement | Proof |
|---|---|
| It sends single `SELECT` statements only. Writes, DDL, `COPY`, `SET`, several statements at once, side-effect functions such as `pg_sleep`, and system tables that hold secrets are refused before they reach your database. | [`test_sql_guard.py::test_every_hostile_query_is_refused_with_its_code`](../../backend/tests/unit/test_sql_guard.py), against Postgres [`test_sql_guard_postgres.py::test_the_hostile_matrix_is_refused_by_postgres`](../../backend/tests/integration/test_sql_guard_postgres.py), CI step "SQL guard against Postgres" |
| It reads the structure: table and column names, types, keys, row counts. | [`test_schema_graph_postgres.py::test_declared_keys_times_and_row_counts_in_postgres`](../../backend/tests/integration/test_schema_graph_postgres.py) |
| It computes column statistics (counts, nulls, distinct values, min, max, mean, percentiles, top values of categorical columns) inside your database, returning one row per group, not your rows. | [`test_db_stats.py::test_no_stats_query_returns_more_than_one_row_per_group`](../../backend/tests/unit/test_db_stats.py), [`test_a_stats_query_is_a_guarded_select`](../../backend/tests/unit/test_db_stats.py), [`test_db_stats_postgres.py`](../../backend/tests/integration/test_db_stats_postgres.py) |
| For a run it copies the tables the task needs (only the wanted columns, only rows dated up to the cutoff) into MLPilot. Rows after the cutoff are not copied. A table without a time column is copied whole. A table over 2,000,000 rows fails the snapshot instead of being cut short. A table that is not in the schema is refused. | [`test_snapshot.py`](../../backend/tests/unit/test_snapshot.py): `test_rows_dated_after_as_of_are_in_the_source_but_not_in_the_snapshot`, `test_a_table_without_a_time_column_is_copied_whole`, `test_a_table_over_the_row_limit_fails_loudly_and_leaves_nothing`, `test_a_snapshot_cannot_name_a_table_outside_the_schema_graph_by_sql_injection` |
| Scoring new entities later reads your live database, again only through the guard. | [`test_sql_guard.py::test_live_sources_run_sql_only_through_the_guard`](../../backend/tests/unit/test_sql_guard.py), [`test_run_score_api.py`](../../backend/tests/integration/test_run_score_api.py) |
| With the role from [`readonly_role.sql`](readonly_role.sql), MLPilot can list and read only the tables you granted. | [`test_pilot_role.py::test_the_role_can_select_the_chosen_tables_and_only_those`](../../backend/tests/integration/test_pilot_role.py), CI job `demo-db` |

## 2. What it stores, and for how long

MLPilot keeps files on its own machine. It has no hosted component.

| Statement | Proof |
|---|---|
| The database password is not stored in clear: either it stays in an environment variable you name, or it is stored encrypted under `MLPILOT_SECRET_KEY`. Without that key MLPilot refuses to store a password. API responses never contain it. | [`test_connections_api.py`](../../backend/tests/integration/test_connections_api.py): `test_create_test_list_delete_a_postgres_connection`, `test_a_password_can_come_from_an_environment_variable_instead`, `test_storing_a_password_without_the_secret_key_is_refused` |
| A snapshot is a copy of the rows described above, in one DuckDB file per project, `backend/data/projects/<project id>/work.duckdb`. Feature values (Parquet cache), trained models, score lists and export bundles are in the same project folder. | The copy: [`test_snapshot.py::test_a_snapshot_is_stored_and_matches_the_source_row_for_row`](../../backend/tests/unit/test_snapshot.py). The folder layout: code only ([`workspace.py`](../../backend/ml/data/workspace.py), [`artifacts.py`](../../backend/ml/export/artifacts.py), `PROJECTS_DIR` in [`datasets.py`](../../backend/app/core/datasets.py)); no test checks the paths. |
| Those files are not encrypted by MLPilot. Use an encrypted disk. | Not tested (there is no encryption code to test). |
| Every prompt sent to a language model, and its answer, is kept in `backend/data/projects/<project id>/prompts/<id>.json`, readable by the owner only. | [`test_privacy_api.py::test_every_prompt_is_logged_with_its_full_text`](../../backend/tests/integration/test_privacy_api.py) |
| The project, task, run, feature (with its SQL), metric and LLM-call records are in a SQLite file, `backend/mlpilot.db` by default (`DATABASE_URL`). | Code only ([`config.py`](../../backend/app/core/config.py)); the schema is checked by [`test_migrations.py`](../../backend/tests/integration/test_migrations.py). |
| **Retention: nothing expires.** No timer deletes snapshots, prompts or scores. They stay until you delete them. | Not tested. A search of the code finds no expiry or deletion job. |

**Deleting everything about a pilot.**

1. On your side: `psql ... -v revoke=drop -f readonly_role.sql` ends the role's sessions and removes the role and its grants. Proof: [`test_pilot_role.py::test_lock_ends_open_sessions_and_logins_and_drop_removes_the_role`](../../backend/tests/integration/test_pilot_role.py).
2. In MLPilot, deleting a saved connection removes its row and its stored secret ([`test_create_test_list_delete_a_postgres_connection`](../../backend/tests/integration/test_connections_api.py)).
3. **`DELETE /api/v1/projects/{id}` removes the project's database rows but not its folder.** Read from [`project_service.py`](../../backend/app/services/project_service.py) (`delete` deletes the row only; nothing in `backend/app` or `backend/ml` removes directories). Not tested. After deleting the project, stop MLPilot and delete `backend/data/projects/<project id>/` by hand. For a clean machine, delete `backend/data/` (it also holds the content-addressed copies of uploaded files under `versions/`), `backend/uploads/` and `backend/mlpilot.db`.

## 3. What a language model is shown

Every prompt is built by one function, the context builder, from the project's privacy setting (the privacy panel in Settings). The default is level 2.

| Level | The prompt contains | Proof |
|---|---|---|
| 1 `schema_only` | Table and column names, types, keys, relationships, event-time columns. No numbers from your data. | [`test_context_builder.py::test_schema_only_has_names_and_types_but_no_numbers_from_the_data`](../../backend/tests/unit/test_context_builder.py) |
| 2 `schema_and_stats` (default) | Level 1 plus aggregates: counts, null shares, distinct counts, min, max, mean, standard deviation, percentiles, first and last timestamp. No category labels, no cell values. **Min, max and the first and last timestamp are real values from your data**; use level 1 if they must not leave. | [`test_context_builder.py::test_schema_and_stats_adds_aggregates_but_no_labels_or_values`](../../backend/tests/unit/test_context_builder.py); default: [`test_privacy_api.py::test_privacy_defaults_and_round_trip`](../../backend/tests/integration/test_privacy_api.py); a scan of every text value of the demo database finds none in a level-2 prompt: [`test_prompt_privacy_demo_db.py::test_schema_and_stats_prompt_holds_no_cell_value`](../../backend/tests/unit/test_prompt_privacy_demo_db.py); same for an uploaded CSV: [`test_privacy_api.py::test_default_prompts_hold_no_value_of_the_uploaded_csv`](../../backend/tests/integration/test_privacy_api.py) |
| 3 `allow_category_labels` | Level 2 plus the most frequent values of categorical columns (for example plan names). Opt in. | [`test_context_builder.py::test_category_labels_are_opt_in`](../../backend/tests/unit/test_context_builder.py), [`test_prompt_privacy_demo_db.py::test_the_scan_does_find_labels_when_the_project_allows_them`](../../backend/tests/unit/test_prompt_privacy_demo_db.py) |
| 4 `allow_sample_values` | The setting exists, is flagged with a warning, and is honoured by the builder, **but nothing supplies example values yet, so it currently adds none.** | [`test_context_builder.py::test_sample_values_are_opt_in_and_flagged`](../../backend/tests/unit/test_context_builder.py); "adds none": code only ([ADR 0009](../adr/0009-privacy-by-default.md)), not tested |

At every level:

- A column marked **never send** is in no prompt, and its name is replaced in free text. Proof: [`test_context_builder.py::test_never_send_column_is_in_no_prompt_at_any_level`](../../backend/tests/unit/test_context_builder.py), [`test_never_send_names_are_replaced_in_free_text_facts_and_system`](../../backend/tests/unit/test_context_builder.py), [`test_prompt_privacy_demo_db.py::test_a_never_send_column_is_in_no_prompt`](../../backend/tests/unit/test_prompt_privacy_demo_db.py).
- What you type (the prediction question, chat messages) is sent as you typed it, apart from the never-send names. MLPilot does not look for personal data in your text. Not tested.
- Nothing is sent that the context builder did not build: the gateway refuses a plain string ([`test_ai_gateway.py::test_gateway_refuses_a_plain_string`](../../backend/tests/unit/test_ai_gateway.py)), and a source scan checks that every call site passes a built prompt ([`test_gateway_call_sites.py`](../../backend/tests/unit/test_gateway_call_sites.py)).
- **With no provider key set, no language model is contacted:** only the offline stub is registered, and it answers locally. Proof: [`test_ai_gateway.py::test_gateway_no_real_providers_when_no_keys`](../../backend/tests/unit/test_ai_gateway.py). A hosted provider is used only if you set its key. What a hosted provider does with the prompt is its own policy; CI never calls a real provider, so those code paths are not tested.
- The model's answer is data. A feature it proposes is parsed, checked by the SQL guard and the point-in-time guard, and tested on held-out time periods before it is kept; the decision is made by code. Proof: [`test_llm_sql.py::test_the_leaky_proposal_is_rejected_by_the_point_in_time_guard`](../../backend/tests/unit/test_llm_sql.py), [`test_free_sql_is_refused_unless_the_project_enables_it`](../../backend/tests/unit/test_llm_sql.py).

**Local model.** Set `OLLAMA_BASE_URL` (see [`.env.example`](../../backend/.env.example)) and route the tiers to `ollama:<model>` in [`config/llm.example.yaml`](../../config/llm.example.yaml). Prompts then go to that URL and cost nothing. Proof: [`test_llm_routing.py::test_ollama_provider_over_mocked_http`](../../backend/tests/unit/test_llm_routing.py) (a mocked HTTP transport) and [`test_example_config_is_valid`](../../backend/tests/unit/test_llm_routing.py). **Not tested:** against a real Ollama server, and that nothing else leaves the machine while a hosted key is also set (a tier that lists a hosted provider can still use it).

## 4. The read-only layers

Any one of these can fail without a write reaching your data.

| Layer | What it does | Proof |
|---|---|---|
| 1. Your grants | The role can `SELECT` the listed tables and nothing else; no insert, update, delete or create. | [`test_pilot_role.py::test_without_the_read_only_default_the_missing_privileges_still_stop_writes`](../../backend/tests/integration/test_pilot_role.py), [`test_the_role_can_select_the_chosen_tables_and_only_those`](../../backend/tests/integration/test_pilot_role.py), CI job `demo-db` |
| 2. Role defaults | New sessions of the role start read-only, with a statement timeout, an idle-in-transaction timeout and a connection limit. A person with the password can change these for their own session; the grants (layer 1) are what hold. | [`test_pilot_role.py`](../../backend/tests/integration/test_pilot_role.py): `test_the_role_cannot_write_with_the_default_session`, `test_the_statement_timeout_cancels_a_long_query`, `test_the_connection_limit_is_enforced`, `test_the_documented_defaults_are_what_the_script_sets` |
| 3. MLPilot checks the role | The connection test reports whether the role could write (superuser, create rights, any write grant) and the UI shows a warning. | [`test_sql_guard_postgres.py::test_the_test_endpoint_reports_can_write_for_a_role_with_insert`](../../backend/tests/integration/test_sql_guard_postgres.py), [`test_pilot_role.py::test_mlpilots_connection_test_calls_the_role_read_only`](../../backend/tests/integration/test_pilot_role.py), [`test_a_write_grant_added_later_is_noticed_and_the_session_default_still_holds`](../../backend/tests/integration/test_pilot_role.py), UI warning: [`frontend/e2e/connect.spec.ts`](../../frontend/e2e/connect.spec.ts) (CI job `demo-db`) |
| 4. Parser | One `SELECT` only (see section 1). | as in section 1 |
| 5. Read-only session | Every statement runs in `START TRANSACTION READ ONLY` with a statement timeout, one statement per call. Even with layer 4 switched off, writes and a second statement are refused, and the session cannot be switched to read-write. | [`test_sql_guard_postgres.py`](../../backend/tests/integration/test_sql_guard_postgres.py): `test_with_the_parser_off_postgres_still_refuses_writes`, `test_with_the_parser_off_postgres_refuses_a_second_statement`, `test_with_the_parser_off_the_session_cannot_be_switched_to_read_write`, `test_statement_and_idle_timeouts_are_set_for_every_session` |
| 6. Result limits | More rows than the limit raise an error; results are never silently cut. | [`test_sql_guard.py::test_row_and_byte_limits_raise_instead_of_truncating`](../../backend/tests/unit/test_sql_guard.py) |

Not covered here: network access to the database (firewall, `pg_hba.conf`, TLS). The connection form has an SSL mode; only the failure case (`ssl_failed`) is tested ([`test_live_sources.py::test_postgres_errors_are_short_fixed_messages`](../../backend/tests/unit/test_live_sources.py)).

## 5. What is logged

| Statement | Proof |
|---|---|
| Each query MLPilot runs is logged as the SHA-256 of its text, its duration, row count and size. The SQL text and the values are not logged. | [`test_sql_guard.py::test_the_query_log_has_a_hash_duration_and_rows_but_never_the_sql_or_values`](../../backend/tests/unit/test_sql_guard.py) |
| Passwords and connection strings are not in the logs, API responses or error messages. A full connect, test and query cycle at DEBUG level is scanned for the password. | [`test_connections_api.py::test_a_full_connect_test_query_cycle_at_debug_level_logs_no_secret`](../../backend/tests/integration/test_connections_api.py), [`test_secrets_and_redaction.py`](../../backend/tests/unit/test_secrets_and_redaction.py), [`test_a_wrong_password_is_reported_without_the_password`](../../backend/tests/integration/test_connections_api.py) |
| Every prompt is written down, with provider, model, size, cost and what it contains, listed by the API (`GET /projects/{id}/llm-calls`). A failed provider attempt is logged too. | [`test_privacy_api.py::test_every_prompt_is_logged_with_its_full_text`](../../backend/tests/integration/test_privacy_api.py), [`test_a_failed_provider_attempt_is_logged_too`](../../backend/tests/integration/test_privacy_api.py) |
| The prompt is logged before it is sent, and a prompt that cannot be logged is not sent. | Not tested. Read from [`gateway.py`](../../backend/ai/gateway.py) (`recorder.started` runs before the provider call and is not caught). |
| Your database's own logs will show MLPilot's statements under the pilot role's name. | Not tested (this is your server's behaviour). |
| MLPilot sends no usage data to its authors. | Not tested. A search of the code finds no telemetry. |
