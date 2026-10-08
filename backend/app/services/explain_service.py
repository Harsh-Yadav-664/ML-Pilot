"""Explanations and grounded answers about a run (#63)."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.gateway import AIGateway
from ai.router import TaskType
from app.core import datasets
from app.db.models import Experiment, Feature, Run
from app.schemas.runs import RunAnswer, RunExplain
from app.services.feature_proposals import feature_description
from app.services.privacy_service import PrivacyService
from ml.export.artifacts import SHAP_FILE, artifact_dir
from ml.reports import explain

logger = logging.getLogger(__name__)
INSTRUCTION = (
    "Answer in at most four sentences using only the records above. Cite the id of every "
    "record you use in square brackets. Do not use any number or name that is not in a record."
)


def _numbers(d: dict[str, Any] | None) -> dict[str, float]:
    return {
        k: v
        for k, v in (d or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


class ExplainService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _run(self, project_id: str, run_id: str) -> Run:
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id:
            raise HTTPException(404, "Run not found in this project")
        return run

    def _shap(self, project_id: str, run_id: str) -> dict[str, Any] | None:
        path = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id) / SHAP_FILE
        return json.loads(path.read_text()) if path.exists() else None  # type: ignore[no-any-return]

    async def shap(self, project_id: str, run_id: str) -> RunExplain:
        await self._run(project_id, run_id)
        stored = self._shap(project_id, run_id)
        if stored is None:
            raise HTTPException(409, "This run has no SHAP values (it ended before they were kept)")
        ranked = sorted(stored["mean_abs"].items(), key=lambda kv: -kv[1])
        return RunExplain(
            method=stored["method"],
            n_rows=stored["n_rows"],
            base_value=stored["base_value"],
            features=[
                {"feature": k, "mean_abs": v, "share": stored["share"][k]} for k, v in ranked
            ],
        )

    async def corpus(self, project_id: str, run_id: str) -> list[explain.Record]:
        run = await self._run(project_id, run_id)
        rows = (
            await self.db.scalars(
                select(Feature).where(Feature.run_id == run_id).order_by(Feature.created_at)
            )
        ).all()
        final = (
            await self.db.get(Experiment, run.champion_experiment_id)
            if run.champion_experiment_id
            else None
        )
        loop = (run.manifest or {}).get("run_loop") or {}
        facts: dict[str, Any] = {
            "status": run.status,
            "stop reason": loop.get("stop_reason"),
            "rounds asked": loop.get("rounds"),
            "features accepted": len(loop.get("accepted") or []),
            "validation": _numbers(final.val_metrics if final else None),
            "test": _numbers(final.test_metrics if final else None),
        }
        features = []
        for f in rows:
            guard = f.guard_results or {}
            features.append(
                {
                    "name": f.name,
                    "kind": f.kind,
                    "status": f.status,
                    "description": feature_description(f),
                    "stage": guard.get("stage"),
                    "reasons": list(guard.get("reasons") or []),
                    "gain": _numbers(f.gain),
                }
            )
        return explain.build_corpus(facts, features, self._shap(project_id, run_id))

    async def ask(
        self, project_id: str, run_id: str, question: str, gateway: AIGateway
    ) -> RunAnswer:
        records = explain.retrieve(question, await self.corpus(project_id, run_id))
        if not records:
            return RunAnswer(answer=explain.NO_RECORD, records=[], mode="no_record")
        ids = [r.id for r in records]
        shown = explain.records_answer(records)
        prompt = (
            (await PrivacyService(self.db).builder(project_id))
            .prompt("runs.ask")
            .text("Question", question)
            .facts("Records", [{"id": r.id, **r.facts} for r in records])
            .text("", INSTRUCTION)
            .build()
        )
        try:
            wording = await gateway.complete(TaskType.ANALYZE, prompt)
        except RuntimeError as e:
            logger.error("The model could not word the answer: %s", e)
            return RunAnswer(
                answer=shown,
                records=ids,
                mode="fallback",
                note=f"The model could not word this ({e}); these are the stored records.",
            )
        bad = explain.violations(wording, records)
        if bad:
            return RunAnswer(
                answer=shown,
                records=ids,
                mode="fallback",
                note="The model's wording used numbers or names that are not in the records "
                f"({', '.join(bad[:5])}), so the records are shown as they are.",
            )
        return RunAnswer(answer=wording, records=ids, mode="llm")
