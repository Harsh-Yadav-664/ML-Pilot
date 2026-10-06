"""The prompt log (#48): every prompt is written down before it is sent.

One `llm_calls` row per attempt (a provider that fails and the next one that answers are two
rows) and one JSON file under `<projects dir>/<project id>/prompts/` with the full system and
user text, the manifest of what the prompt contains, and the response. The file is the only
place the text is kept; the row holds a hash, sizes, cost and the path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import LLMResult, SentPrompt
from app.db.models.llm_call import LLMCall

UNASSIGNED = "_no_project"


def prompt_file(projects_dir: Path, project_id: str | None, row_id: str) -> Path:
    return projects_dir / (project_id or UNASSIGNED) / "prompts" / f"{row_id}.json"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


class PromptLog:
    """A `CallRecorder` that stores prompts in the metadata database and on disk."""

    def __init__(self, sessions: Callable[[], AsyncSession], projects_dir: Path) -> None:
        self.sessions = sessions
        self.projects_dir = projects_dir

    @staticmethod
    def row_id(sent: SentPrompt) -> str:
        return f"{sent.prompt_id}-{sent.provider}"

    def _payload(
        self, sent: SentPrompt, result: LLMResult | None, response: Any, error: str | None
    ) -> dict[str, Any]:
        built = sent.built
        return {
            "id": self.row_id(sent),
            "prompt_id": sent.prompt_id,
            "project_id": built.project_id,
            "purpose": built.purpose,
            "task_type": sent.task_type,
            "provider": sent.provider,
            "model": sent.model,
            "created_at": datetime.now(UTC).isoformat(),
            "system": built.system,
            "prompt": built.text,
            "manifest": asdict(built.manifest),
            "response": response,
            "error": error,
            "result": result.meta() if result else None,
        }

    async def started(self, sent: SentPrompt) -> None:
        built = sent.built
        row_id = self.row_id(sent)
        path = prompt_file(self.projects_dir, built.project_id, row_id)
        _write(path, self._payload(sent, None, None, None))
        digest = hashlib.sha256(f"{built.system}\n{built.text}".encode()).hexdigest()
        async with self.sessions() as db:
            db.add(
                LLMCall(
                    id=row_id,
                    project_id=built.project_id,
                    purpose=built.purpose,
                    provider=sent.provider,
                    model=sent.model,
                    decision_mode="pending",
                    privacy_level=built.manifest.level,
                    prompt_hash=digest,
                    prompt_path=str(path.relative_to(self.projects_dir)),
                    prompt_chars=len(built.system) + len(built.text),
                )
            )
            await db.commit()

    async def finished(
        self, sent: SentPrompt, result: LLMResult | None, response: Any, error: str | None
    ) -> None:
        row_id = self.row_id(sent)
        _write(
            prompt_file(self.projects_dir, sent.built.project_id, row_id),
            self._payload(sent, result, response, error),
        )
        async with self.sessions() as db:
            row = await db.get(LLMCall, row_id)
            if row is None:
                raise LookupError(f"llm_calls row {row_id} vanished while the call ran")
            if result is not None:
                row.decision_mode = result.decision_mode
                row.tokens_in = result.tokens_in
                row.tokens_out = result.tokens_out
                row.cost_usd = result.cost_usd
                row.latency_ms = result.latency_ms
            else:
                row.decision_mode = "failed"
                row.error = error
            await db.commit()
