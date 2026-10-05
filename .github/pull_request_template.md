Fixes #

## What changed
<!-- Before / after, in plain words. -->

## Proof for each acceptance criterion
<!-- Copy every acceptance criterion from the issue. Under each one, paste the test name, CI link or command output that shows it passing. -->
- [ ] criterion: proof

## Rules check (AGENTS.md section 2)
- [ ] No LLM output is executed as code; SQL goes through the guards
- [ ] No writes to user databases; no secrets in code, logs or responses
- [ ] No tuning or selection on the test set; no future data in features
- [ ] No fake or simulated numbers outside Demo mode
- [ ] `pytest`, `ruff check .` and `npm run build` pass locally

## Notes / assumptions
