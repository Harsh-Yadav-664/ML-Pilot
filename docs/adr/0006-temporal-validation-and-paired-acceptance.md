# 0006. Time-based validation and a paired acceptance rule

**Status:** Accepted. The acceptance rule and the train/validation/test discipline are implemented for single-table tasks (random splits). Time-based splits for relational tasks are implemented (#52) and the relational run loop (#58) accepts features by the same paired rule on time-ordered folds of the training rows (`ml/features/gain.py`).

## Context

Two ways a tool can fool itself: tuning on the data it reports, and keeping a feature because of noise. Churn-style problems are also about the future, so a random split lets the model see the future.

## Decision

- **One split contract** (`ml/validation/splits.py`): tuning and every decision use training and validation rows only; the test rows are scored once, at the end. Small data uses inner K-fold CV instead of a holdout.
- **Acceptance by paired comparison.** A candidate feature is kept only if its mean gain over the current feature set, measured on the same repeated K-fold splits of the training rows, beats a margin tied to the noise (`ml/experiments/acceptance.py`). The test rows never take part.
- **Relational tasks split by cutoff date**, never at random (#52). Today's single-table tasks still use random stratified splits, so their numbers say nothing about performance on future data.
- The threshold and any probability calibration are fitted on validation predictions and applied unchanged to test.

## Consequences

- Every candidate costs repeated model fits; the budget (#57) and a fast acceptance model keep this tolerable.
- A real but small gain can be rejected. That is the intended direction of the error.

## Alternatives considered

- **Compare a single validation score.** Too noisy; accepts luck.
- **Let the LLM judge the numbers.** See 0001.

## Evidence and issues

`test_honest_split.py`, `test_executor_splits.py`, `test_acceptance.py`, `test_threshold_and_calibration_split.py`. Issues: #34, #35, #93, #52 (time-based, planned).
