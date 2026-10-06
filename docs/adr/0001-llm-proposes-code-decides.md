# 0001. The LLM proposes; deterministic code validates, executes and decides

**Status:** Accepted. Implemented for formula features; the SQL parts are `planned (#56)`.

## Context

An LLM is good at suggesting features and bad at being trusted: it can invent columns, write code that does something else, or talk itself into keeping a feature that does not help. A tool that holds database credentials cannot run whatever a model writes.

## Decision

LLM output is data. It is parsed into a structured object, checked, and only then used by code we wrote.

- Feature formulas go through one whitelist evaluator (`ml/features/safe_eval.py`). An invalid formula rejects the experiment; it never becomes a column of zeros.
- Nothing from a model is passed to `eval`, `exec`, a shell or a database without a guard.
- Whether a feature is kept is decided by a statistical rule in code (`ml/experiments/acceptance.py`, see 0006). The LLM may explain the decision afterwards, and if that call fails the decision stands (`explanation_mode: fallback`).
- When an LLM call falls back to the offline stub, the result is marked `decision_mode: fallback` and shown in the UI.

## Consequences

- Features are limited to what the evaluator and, later, the feature spec can express. That is a deliberate limit.
- Every LLM-facing code path needs a stub-provider test, so tests need no network or key.
- Some good ideas will be rejected because they are not expressible. We accept that over running unreviewed code.

## Alternatives considered

- **Let the LLM write Python and run it in a sandbox.** Sandboxes are hard to get right and the output is hard for a user to review. Rejected.
- **Let the LLM decide keep or reject.** It agrees with itself too easily and the decision cannot be reproduced. Rejected.

## Evidence and issues

`test_safe_eval.py`, `test_invalid_formula.py`, `test_acceptance.py`, `test_agent_loop_telecom.py`. Issues: #37, #35, #56 (SQL proposer, planned).
