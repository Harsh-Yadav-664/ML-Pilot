# 0009. The LLM sees schema and aggregates, not rows

**Status:** Implemented (#48): `backend/ai/context_builder.py`, the prompt log and four privacy levels; tests in `test_context_builder.py`, `test_prompt_privacy_demo_db.py`, `test_privacy_api.py`.

## Context

Customer databases contain personal and commercial data. Sending raw rows to a hosted LLM is a decision the data owner must make, not a default.

## Decision

By default an LLM prompt contains table and column names, types, relationships and aggregate statistics (counts, ranges, missing rates, top-level distributions that cannot identify a person), and never raw cell values. All prompts are built by one context builder, which logs every prompt with a hash so what left the machine can be audited. A project can raise or lower the level; the highest level is a local model (Ollama) so nothing leaves the machine.

Levels, each adding to the previous: `schema_only`; `schema_and_stats` (the default: counts, null shares, distinct counts, and min, max, mean, standard deviation and percentiles of numeric columns, first and last timestamp of time columns; no category labels); `allow_category_labels`; `allow_sample_values` (shows a warning). Min and max are real values by nature, so a project that cannot show them uses `schema_only`. Columns marked `never_send` are left out at every level and their names are replaced in free text and facts.

`AIGateway` accepts only a `BuiltPrompt`, which only `ContextBuilder` creates; a plain string raises `TypeError`. The prompt log (`llm_calls` row plus a JSON file under the project directory) is written before a prompt is sent, so a prompt that cannot be logged is not sent, and again with the response or the error.

## Consequences

- The LLM cannot see an example value, so some feature ideas will be less well informed. Aggregates carry most of what it needs.
- All prompt construction must go through one place; code that builds a prompt on its own is a bug. The gate is checked when the call is made, not by the type checker alone.
- `allow_sample_values` is accepted and honoured by the builder, but nothing produces example values yet, so it currently adds none.
- Statistics reach the prompt through a fixed list of names; a new statistic has to be added there on purpose.

## Alternatives considered

- **Send sample rows to improve suggestions.** Better ideas, unacceptable as a default.
- **Redact values after the fact.** Easy to miss one. Rejected as the primary control.

## Issues

#48 (controls and prompt log, implemented), #96 (aggregate-only column statistics, implemented), #41 (local Ollama provider, implemented).
