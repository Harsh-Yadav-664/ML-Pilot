# 0004. The point-in-time guard analyses the SQL, instead of trusting prompt rules

**Status:** Accepted. `planned (#51, #54)`: there is no relational feature code yet. Single-table runs use a different safeguard (the leakage scanner, #38).

## Context

The costliest mistake in this kind of tool is a feature that uses data from after the prediction date. The model then looks excellent in testing and fails in production. Telling the LLM "only use past data" in the prompt is not a control: it will sometimes not comply.

## Decision

Every feature query is parsed and checked before it runs. Each table with an event time must be filtered to rows before the row's cutoff date. The guard rewrites or rejects the query so a feature that ignores the cutoff cannot execute. The check is code, not instructions to a model.

To prove it works, CI plants future data in test databases (canaries) and requires that every feature touching it is blocked or rewritten (#54).

## Consequences

- We must know which column is each table's event time (#45 infers and lets the user correct it).
- Some queries cannot be proven safe and will be rejected.
- The guard becomes critical code: it has the most tests and gets the closest review.

## Alternatives considered

- **Prompt rules only.** Not enforceable. Rejected.
- **Build the training table row by row with per-row filters.** Slow, and it still needs the same analysis.
- **Only the leakage scanner after the fact.** Finds some leaks, misses others. Kept as a second layer, not a replacement.

## Issues

#51 (guard), #50 (labels at cutoffs), #54 (canaries), #45 (time columns), #38 (single-table leakage scanner, implemented).
