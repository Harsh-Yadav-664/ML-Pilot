# 0009. The LLM sees schema and aggregates, not rows

**Status:** Accepted as a rule (AGENTS.md rule 6). The central context builder, the prompt log and the privacy levels are `planned (#48, #96)`. Today's prompts are built in several places and are not yet audited against this rule.

## Context

Customer databases contain personal and commercial data. Sending raw rows to a hosted LLM is a decision the data owner must make, not a default.

## Decision

By default an LLM prompt contains table and column names, types, relationships and aggregate statistics (counts, ranges, missing rates, top-level distributions that cannot identify a person), and never raw cell values. All prompts are built by one context builder, which logs every prompt with a hash so what left the machine can be audited. A project can raise or lower the level; the highest level is a local model (Ollama) so nothing leaves the machine.

## Consequences

- The LLM cannot see an example value, so some feature ideas will be less well informed. Aggregates carry most of what it needs.
- All prompt construction must go through one place; code that builds a prompt on its own is a bug.

## Alternatives considered

- **Send sample rows to improve suggestions.** Better ideas, unacceptable as a default.
- **Redact values after the fact.** Easy to miss one. Rejected as the primary control.

## Issues

#48 (controls and prompt log), #96 (aggregate-only column statistics), #41 (local Ollama provider, implemented).
