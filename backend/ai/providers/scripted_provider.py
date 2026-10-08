"""ScriptedProvider: replays fixed structured answers, for the end-to-end test only.

Set ``MLPILOT_TEST_SCRIPTED_LLM=<answers.yaml>:<key1>,<key2>,...`` and the gateway puts this
provider first for feature proposals. Each call returns the next named answer from the YAML file
(the same file the unit tests use: tests/fixtures/stub_feature_proposals.yaml). Nothing is made up
here: the answers are what a model might say, and every check on them (schema, SQL guard,
point-in-time guard, gain test) is the real one. Records show the provider as ``scripted``, never
as a real model. When the script runs out it raises, so the gateway falls back to the stub and the
run records ``no_llm``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ml.core.interfaces import AIProvider, ModelInfo

ENV_VAR = "MLPILOT_TEST_SCRIPTED_LLM"
MODEL = "scripted-v1"


class ScriptedProvider(AIProvider):
    name = "scripted"
    default_model = MODEL

    def __init__(self, spec: str) -> None:
        path, _, keys = spec.rpartition(":")
        if not path or not keys:
            raise ValueError(f"{ENV_VAR} must look like '<answers.yaml>:<key>,<key>', got {spec!r}")
        answers = yaml.safe_load(Path(path).read_text())
        self._queue: list[dict[str, Any]] = [copy.deepcopy(answers[k]) for k in keys.split(",")]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        raise RuntimeError("the scripted provider only answers structured requests")

    async def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        if not self._queue:
            raise RuntimeError("the script has no more answers")
        return self._queue.pop(0)

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str | None = None
    ) -> float:
        return 0.0

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name=MODEL,
                provider=self.name,
                context_window=8192,
                cost_per_1k_prompt_tokens=0.0,
                cost_per_1k_completion_tokens=0.0,
                capabilities=["structured"],
                is_free=True,
            )
        ]

    async def health_check(self) -> bool:
        return True
