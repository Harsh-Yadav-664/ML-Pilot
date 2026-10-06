# 0007. LightGBM by default, optional engines, permissive licences only

**Status:** Accepted. Implemented in `ml/models/engines/`.

## Context

A good default model must be fast, strong on tabular data and free for commercial use. Some attractive models have licences that forbid exactly the use this project is for.

## Decision

- LightGBM is the default engine. XGBoost, scikit-learn models, and the optional AutoGluon and TabICL sit behind one engine interface and are used only if installed.
- Every dependency and default model must have a permissive licence (MIT, BSD, Apache-2.0, PostgreSQL). Non-commercial or "not for production" packages and weights are not defaults and not bundled.
- Every run records the engine name and version, so results can be reproduced (the run manifest).

## Consequences

- A model that is slightly better but not freely usable is out of the default path.
- Optional engines mean an engine can be missing; asking for one that is not installed fails with a clear error, not a silent swap.

## Alternatives considered

- **AutoGluon as the default.** Strong, but heavy to install and slow for one run per feature. Kept as an option, and as the benchmark to beat (#67).
- **A neural tabular model as the default.** Slower and less predictable. Rejected for now.

## Evidence and issues

`test_engines.py`, `test_engine_e2e.py`, `test_rerun.py`. Issues: #40, #94, #67.
