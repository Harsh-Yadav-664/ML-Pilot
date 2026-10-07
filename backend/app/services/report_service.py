"""The evidence report of a run (#60): read what the run stored, hand it to ``ml.reports``.

``records`` is the only thing the report may say. It is plain JSON built from stored rows; the
two parts that are recomputed from the snapshot (label balance) or read from the connection
(schema scan) say so, and when they cannot be read the report says why instead of leaving a gap.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.context_builder import LEVEL_SUMMARY
from app.db.models import Connection, DataVersion, Experiment, Feature, Run, TaskSpec
from app.schemas.tasks import LabelPreviewRequest
from app.services.connection_service import ConnectionService
from app.services.feature_proposals import feature_description
from app.services.privacy_service import PrivacyService
from app.services.task_service import TaskService
from ml.data.sources import ConnectionFailure
from ml.reports import Report, build_report, to_html, to_markdown
from ml.tasks.nl_to_spec import describe_spec
from ml.tasks.spec import from_yaml

ReportFormat = Literal["markdown", "html", "json"]


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _day(value: Any) -> str:
    return str(value)[:10]


class ReportService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _run(self, project_id: str, run_id: str) -> Run:
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id or run.task_spec_id is None:
            raise HTTPException(404, "Run not found in this project")
        return run

    async def records(self, project_id: str, run_id: str) -> dict[str, Any]:
        run = await self._run(project_id, run_id)
        spec_row = await self.db.get(TaskSpec, run.task_spec_id)
        assert spec_row is not None
        spec = from_yaml(spec_row.yaml)
        experiments = list(
            (
                await self.db.scalars(
                    select(Experiment)
                    .where(Experiment.run_id == run_id)
                    .order_by(Experiment.created_at)
                )
            ).all()
        )
        kinds = {
            k: [e for e in experiments if (e.manifest or {}).get("kind") == k]
            for k in ("baseline", "champion", "final")
        }
        if not kinds["baseline"]:
            raise HTTPException(409, "The run has no baseline yet: there is nothing to report")
        baseline, final = kinds["baseline"][0], (kinds["final"][0] if kinds["final"] else None)
        features = list(
            (
                await self.db.scalars(
                    select(Feature).where(Feature.run_id == run_id).order_by(Feature.created_at)
                )
            ).all()
        )
        proposals = [f for f in features if f.kind == "llm_sql"]
        dfs = [f for f in features if f.kind == "dfs"]
        dfs.sort(key=lambda f: -float((f.gain or {}).get("importance") or 0.0))
        calls = [
            {**c, "log_id": f"{c['prompt_id']}-{c['provider']}"}  # the prompt log's row id
            for f in proposals
            for c in ((f.guard_results or {}).get("llm") or [])
        ]
        tested = [f.gain for f in proposals if f.gain]
        plan = run.split_plan or {}
        draft = spec_row.draft_source or {}
        data = await self._data(run)
        if data["id"] is None:
            raise HTTPException(409, "The run has no data version: there is nothing to report")
        statuses = [f.status for f in proposals]
        records: dict[str, Any] = {
            "run": {
                "id": run.id,
                "status": run.status,
                "error": run.error,
                "engine": run.engine,
                "seed": run.seed,
                "split_plan": {
                    "val_from": _day(plan.get("val_from")),
                    "test_from": _day(plan.get("test_from")),
                    "folds": plan.get("folds"),
                },
                "as_of": (run.manifest or {}).get("as_of"),
                "finished_at": _iso(run.finished_at),
                "budget": run.budget or {},
                "budget_used": run.budget_used or {},
                "manifest": run.manifest or {},
            },
            "task": {
                "name": spec_row.name,
                "version": spec_row.version,
                "status": spec_row.status,
                "confirmed_by": spec_row.confirmed_by,
                "confirmed_at": _iso(spec_row.confirmed_at),
                "yaml": spec_row.yaml,
                "words": describe_spec(spec),
                "entity_table": spec.entity.table,
                "question": draft.get("question"),
                "decision_mode": draft.get("decision_mode"),
                "assumptions": list(draft.get("assumptions") or []),
            },
            "data": data,
            "baseline": {
                "experiment": {
                    "val_metrics": baseline.val_metrics,
                    "engine": (baseline.manifest or {}).get("engine"),
                    "engine_version": (baseline.manifest or {}).get("engine_version"),
                    "params": (baseline.manifest or {}).get("params"),
                },
                "info": (run.manifest or {}).get("baseline") or {},
                "features": [
                    {
                        "name": f.name,
                        "description": feature_description(f) or f.name,
                        "importance": (f.gain or {}).get("importance"),
                    }
                    for f in dfs
                    if (f.gain or {}).get("importance") is not None
                ],
            },
            "final": (
                {
                    "model_name": final.model_name,
                    "val_metrics": final.val_metrics,
                    "test_metrics": final.test_metrics,
                    "feature_set": final.feature_set,
                    "manifest": final.manifest or {},
                }
                if final is not None
                else None
            ),
            "features": [
                {
                    "name": f.name,
                    "status": f.status,
                    "sql": f.sql,
                    "description": feature_description(f) or f.name,
                    "gain": f.gain,
                    "guard_results": f.guard_results or {},
                }
                for f in proposals
            ],
            "champions": [
                {
                    "round": (e.manifest or {}).get("round"),
                    "feature": (e.change_description or "").removeprefix("Added feature "),
                    "val_pr_auc": (e.val_metrics or {}).get("pr_auc"),
                }
                for e in sorted(
                    kinds["champion"], key=lambda e: (e.manifest or {}).get("round") or 0
                )
            ],
            "acceptance": (
                {
                    "n_folds": tested[0]["rule"].get("folds"),
                    "min_gain": tested[0]["rule"].get("min_gain"),
                    "std_multiplier": tested[0]["rule"].get("std_multiplier"),
                }
                if tested
                else None
            ),
            "experiments": {
                "total": len(experiments),
                "with_test_metrics": sum(e.test_metrics is not None for e in experiments),
            },
            "summary": {
                "proposed": len(proposals),
                "accepted": statuses.count("accepted"),
                "rejected": len(proposals) - statuses.count("accepted"),
                "refused": statuses.count("rejected_guard") + statuses.count("rejected_duplicate"),
                "vetoed": statuses.count("vetoed"),
                "n_features": len(final.feature_set) if final is not None else None,
            },
            "llm": {
                "calls": calls,
                "n_calls": len(calls),
                "tokens_in": sum(c.get("tokens_in", 0) for c in calls),
                "tokens_out": sum(c.get("tokens_out", 0) for c in calls),
                "cost_usd": sum(c.get("cost_usd", 0.0) for c in calls),
                "n_fallback": sum(c.get("decision_mode") == "fallback" for c in calls),
            },
            "derived": {},
        }
        test = final.test_metrics if final is not None else None
        if test and test.get("n_test") is not None and test.get("base_rate") is not None:
            records["derived"]["test_positives"] = round(test["n_test"] * test["base_rate"])
        records["privacy"] = await self._privacy(project_id)
        records["connection"] = await self._connection(spec_row)
        records["schema"] = await self._schema(spec_row)
        records["labels"] = await self._labels(project_id, spec_row, run)
        return records

    async def _data(self, run: Run) -> dict[str, Any]:
        version = (
            await self.db.get(DataVersion, run.data_version_id) if run.data_version_id else None
        )
        if version is None:
            return {"id": None, "kind": "unknown", "n_rows": None, "tables": []}
        tables = (version.source or {}).get("tables") or {}
        return {
            "id": version.id,
            "kind": version.kind,
            "n_rows": version.n_rows,
            "tables": [
                {
                    "name": name,
                    "rows": t.get("rows"),
                    "max_event_time": t.get("max_event_time"),
                    "null_time_rows": t.get("null_time_rows"),
                }
                for name, t in sorted(tables.items())
            ],
        }

    async def _privacy(self, project_id: str) -> dict[str, Any]:
        policy = await PrivacyService(self.db).policy(project_id)
        return {
            "level": policy.level.value,
            "summary": LEVEL_SUMMARY[policy.level],
            "never_send": sorted(policy.never_send, key=str.lower),
        }

    async def _connection(self, spec_row: TaskSpec) -> dict[str, Any]:
        conn = (
            await self.db.get(Connection, spec_row.connection_id)
            if spec_row.connection_id
            else None
        )
        if conn is None:
            return {"available": False, "reason": "The task's connection no longer exists."}
        return {"available": True, "dialect": conn.dialect, "can_write": conn.can_write}

    async def _schema(self, spec_row: TaskSpec) -> dict[str, Any]:
        conn = (
            await self.db.get(Connection, spec_row.connection_id)
            if spec_row.connection_id
            else None
        )
        if conn is None:
            return {"available": False, "reason": "The task's connection no longer exists."}
        try:
            graph = await ConnectionService(self.db).schema_graph(conn)
        except (HTTPException, ConnectionFailure) as e:
            reason = e.detail if isinstance(e, HTTPException) else str(e)
            return {"available": False, "reason": f"The schema could not be read now: {reason}"}
        return {
            "available": True,
            "warnings": list(graph.warnings),
            "tables": [
                {
                    "key": t.key,
                    "is_static": t.is_static,
                    "leakage_hint": t.time_leakage_hint,
                    "mutable": [
                        {"name": c.name, "source": c.mutable_source or "name"}
                        for c in t.columns
                        if c.mutable
                    ],
                }
                for t in graph.tables
            ],
        }

    async def _labels(self, project_id: str, spec_row: TaskSpec, run: Run) -> dict[str, Any]:
        try:
            preview = await TaskService(self.db).preview_labels(
                project_id, spec_row.id, LabelPreviewRequest(data_version_id=run.data_version_id)
            )
        except HTTPException as e:
            return {
                "available": False,
                "reason": f"The labels could not be rebuilt now: {e.detail}",
            }
        return {
            "available": True,
            "sql": preview.sql,
            "cutoffs": [
                {
                    "cutoff": c.cutoff.isoformat(),
                    "eligible": c.eligible,
                    "positives": c.positives,
                    "base_rate": c.base_rate,
                }
                for c in preview.cutoffs
            ],
        }

    async def report(self, project_id: str, run_id: str) -> Report:
        return build_report(await self.records(project_id, run_id))

    async def render(self, project_id: str, run_id: str, fmt: ReportFormat) -> tuple[str, str]:
        """The report as (text, media type)."""
        if fmt == "json":
            records = await self.records(project_id, run_id)
            report = build_report(records)
            return json.dumps(
                {"records": records, "numbers": report.numbers, "verbatim": report.verbatim},
                default=str,
                indent=2,
            ), "application/json"
        report = await self.report(project_id, run_id)
        if fmt == "html":
            return to_html(report), "text/html; charset=utf-8"
        return to_markdown(report), "text/markdown; charset=utf-8"
