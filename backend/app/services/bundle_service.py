"""The export bundle of a finished run (#61)."""

from __future__ import annotations

import asyncio
import json
from importlib import metadata
from typing import Literal

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import Connection, Experiment, Feature, Run, TaskSpec
from app.services.connection_service import ConnectionService
from app.services.feature_proposals import feature_description
from app.services.report_service import ReportService
from ml.export.artifacts import CATEGORIES_FILE, MODEL_FILE, REFERENCE_FILE, artifact_dir
from ml.export.bundle import (
    DIALECTS,
    BundleInput,
    ExportError,
    ExportFeature,
    build_bundle,
    zip_bundle,
)
from ml.tasks.spec import from_yaml

Dialect = Literal["duckdb", "postgres"]
PINNED = ("lightgbm", "numpy", "pandas", "scikit-learn", "sqlglot", "duckdb", "pg8000")


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in PINNED}


class BundleService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def bundle(
        self, project_id: str, run_id: str, dialect: Dialect | None = None
    ) -> tuple[bytes, str]:
        """(zip, file name). 409 if the run has not ended or kept no model."""
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id or run.task_spec_id is None:
            raise HTTPException(404, "Run not found in this project")
        if run.status not in ("completed", "stopped") or run.champion_experiment_id is None:
            raise HTTPException(409, f"A run in status '{run.status}' has nothing to export yet")
        folder = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id)
        if not all((folder / n).exists() for n in (MODEL_FILE, REFERENCE_FILE, CATEGORIES_FILE)):
            raise HTTPException(409, "This run kept no model file (it ended before export existed)")
        spec_row = await self.db.get(TaskSpec, run.task_spec_id)
        conn = await self.db.get(Connection, spec_row.connection_id) if spec_row else None
        if spec_row is None or conn is None:
            raise HTTPException(409, "The task or its connection no longer exists")
        champion = await self.db.get(Experiment, run.champion_experiment_id)
        assert champion is not None
        notes: list[str] = []
        target = dialect or (conn.dialect if conn.dialect in DIALECTS else "duckdb")
        if dialect is None and conn.dialect not in DIALECTS:
            notes.append(
                f"The source database is {conn.dialect}; the queries are written for DuckDB, "
                "so load the tables into DuckDB (or ask for dialect=postgres) to run them."
            )
        rows = (
            await self.db.scalars(
                select(Feature)
                .where(Feature.run_id == run_id, Feature.sql.is_not(None))
                .order_by(Feature.created_at.desc())
            )
        ).all()
        by_name: dict[str, Feature] = {}
        for f in rows:
            by_name.setdefault(f.name, f)
        missing = [n for n in champion.feature_set if n not in by_name]
        if missing:
            raise HTTPException(409, f"The SQL of {len(missing)} champion features is missing")
        graph = await ConnectionService(self.db).schema_graph(conn)
        text, _ = await ReportService(self.db).render(project_id, run_id, "html")
        reference = pd.read_csv(folder / REFERENCE_FILE)
        try:
            files = await asyncio.to_thread(
                build_bundle,
                BundleInput(
                    run_id=run_id,
                    spec=from_yaml(spec_row.yaml),
                    graph=graph,
                    dialect=target,
                    features=[
                        ExportFeature(n, feature_description(by_name[n]) or n, str(by_name[n].sql))
                        for n in champion.feature_set
                    ],
                    model_text=(folder / MODEL_FILE).read_text(),
                    reference=reference,
                    validation=dict(champion.val_metrics or {}),
                    test=dict(champion.test_metrics) if champion.test_metrics else None,
                    manifest=dict(run.manifest or {}),
                    versions=_versions(),
                    categories=json.loads((folder / CATEGORIES_FILE).read_text()),
                    report_html=text,
                    notes=notes,
                ),
            )
        except ExportError as e:
            raise HTTPException(422, str(e)) from e
        name = f"mlpilot_export_{run_id[:8]}"
        return zip_bundle(files, name), f"{name}.zip"
