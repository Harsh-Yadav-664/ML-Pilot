"""Where a loop reports progress and checks for cancellation (``app/jobs`` JobContext)."""

from __future__ import annotations

from typing import Any, Protocol


class LoopHooks(Protocol):
    """Where the loop reports progress and checks for cancellation (app/jobs JobContext)."""

    async def step(self, name: str, progress: float | None = None, **payload: Any) -> None: ...

    async def emit(self, type_: str, **payload: Any) -> int: ...

    async def check_cancelled(self) -> None: ...


class NoHooks:
    """Used when the loop runs outside the job runner (scripts, unit tests)."""

    async def step(self, name: str, progress: float | None = None, **payload: Any) -> None:
        return None

    async def emit(self, type_: str, **payload: Any) -> int:
        return 0

    async def check_cancelled(self) -> None:
        return None
