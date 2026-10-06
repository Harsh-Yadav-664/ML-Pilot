"""One pattern for failures that must not stop the work but must stay visible (AGENTS.md rule 8).

A failure has exactly three allowed outcomes:

1. Expected and recoverable (e.g. ROC-AUC is undefined when only one class is present):
   ``unavailable()`` sets the value to ``None`` and records why in a notes dict.
2. Part of a run fails (tuning, the ensemble, one feature): ``step_failed()`` marks the
   step ``failed (<message>)`` on the run record and lists what was skipped as a result.
3. A request fails: raise ``HTTPException`` with a clear message (no secrets).

Nothing else: no ``except: pass`` and no empty result standing in for an error.
This module has no app dependencies so ``ml/`` code can use it too.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def describe(exc: BaseException) -> str:
    """A short message for an exception; falls back to its type when it has no text."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def unavailable(values: dict[str, Any], notes: dict[str, str] | None, name: str, reason: str) -> None:
    """Outcome 1: record ``name`` as ``None`` with the reason it has no value."""
    values[name] = None
    if notes is not None:
        notes[name] = reason


def step_failed(record: dict[str, Any], step: str, exc: BaseException, skipped: str | None = None) -> str:
    """Outcome 2: mark ``step`` as ``failed (<message>)`` on ``record`` and say what was skipped."""
    status = f"failed ({describe(exc)})"
    record[step] = status
    if skipped:
        record.setdefault("skipped", []).append(f"{step}: {skipped}")
    logger.warning("Step %r failed: %s", step, describe(exc), exc_info=exc)
    return status
