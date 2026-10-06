"""CostTracker — logs per-request AI usage and cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class UsageRecord:
    """A single AI API usage record."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    task_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    experiment_id: str | None = None
    project_id: str | None = None


class CostTracker:
    """Track AI API usage and cost across all providers."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def log(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        task_type: str = "unknown",
        experiment_id: str | None = None,
        project_id: str | None = None,
    ) -> UsageRecord:
        """Log a usage record and return it."""
        record = UsageRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            task_type=task_type,
            experiment_id=experiment_id,
            project_id=project_id,
        )
        self._records.append(record)
        return record

    def total_cost(self, provider: str | None = None) -> float:
        """Return total cost in USD, optionally filtered by provider."""
        records = self._records
        if provider:
            records = [r for r in records if r.provider == provider]
        return sum(r.cost_usd for r in records)

    def total_tokens(self) -> dict[str, int]:
        """Return total prompt and completion token counts."""
        return {
            "prompt_tokens": sum(r.prompt_tokens for r in self._records),
            "completion_tokens": sum(r.completion_tokens for r in self._records),
        }

    def get_records(
        self,
        provider: str | None = None,
        project_id: str | None = None,
    ) -> list[UsageRecord]:
        """Return usage records, optionally filtered."""
        records = self._records
        if provider:
            records = [r for r in records if r.provider == provider]
        if project_id:
            records = [r for r in records if r.project_id == project_id]
        return records

    def summary(self) -> dict:
        """Return a summary of all usage."""
        by_provider: dict[str, dict] = {}
        for r in self._records:
            if r.provider not in by_provider:
                by_provider[r.provider] = {
                    "requests": 0,
                    "cost_usd": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            by_provider[r.provider]["requests"] += 1
            by_provider[r.provider]["cost_usd"] += r.cost_usd
            by_provider[r.provider]["prompt_tokens"] += r.prompt_tokens
            by_provider[r.provider]["completion_tokens"] += r.completion_tokens
        return {
            "total_requests": len(self._records),
            "total_cost_usd": self.total_cost(),
            "by_provider": by_provider,
        }
