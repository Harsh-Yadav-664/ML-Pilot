"""Score the entities eligible at a cutoff with a finished run's champion (#62)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import datasets
from app.db.models import Connection, DataVersion, Experiment, Run, TaskSpec
from app.schemas.runs import RunScore
from app.services.bundle_service import latest_features
from app.services.connection_service import ConnectionService
from ml.data import sql_guard
from ml.data.engine import EngineError
from ml.data.schema_graph import SchemaGraph
from ml.data.snapshot import DataDescription
from ml.data.sources import ConnectionFailure
from ml.export.artifacts import (
    CATEGORIES_FILE,
    MODEL_FILE,
    STATS_FILE,
    artifact_dir,
)
from ml.scoring import score as scoring
from ml.tasks.spec import from_yaml

QUERY_TIMEOUT_S = 600
MAX_ENTITIES = 1_000_000
PREVIEW_ROWS = 25


def csv_path(folder: Path, cutoff: datetime) -> Path:
    return folder / "scores" / f"{cutoff:%Y%m%dT%H%M%S}.csv"


class ScoreService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def score(
        self, project_id: str, run_id: str, cutoff: datetime | None, top_k: int
    ) -> RunScore:
        run = await self.db.get(Run, run_id)
        if run is None or run.project_id != project_id or run.task_spec_id is None:
            raise HTTPException(404, "Run not found in this project")
        if run.status not in ("completed", "stopped") or run.champion_experiment_id is None:
            raise HTTPException(409, f"A run in status '{run.status}' cannot score yet")
        folder = artifact_dir(datasets.PROJECTS_DIR, project_id, run_id)
        if not all((folder / n).exists() for n in (MODEL_FILE, CATEGORIES_FILE, STATS_FILE)):
            raise HTTPException(
                409, "This run kept no model file (it ended before scoring existed)"
            )
        spec_row = await self.db.get(TaskSpec, run.task_spec_id)
        conn = await self.db.get(Connection, spec_row.connection_id) if spec_row else None
        version = (
            await self.db.get(DataVersion, run.data_version_id) if run.data_version_id else None
        )
        champion = await self.db.get(Experiment, run.champion_experiment_id)
        if spec_row is None or conn is None or version is None or champion is None:
            raise HTTPException(409, "The task, connection or data version no longer exists")
        connections = ConnectionService(self.db)
        graph = await connections.schema_graph(conn)
        stored = await latest_features(self.db, run_id)
        names = list(champion.feature_set)
        features = {n: str(stored[n].sql) for n in names if n in stored}
        if len(features) != len(names):
            raise HTTPException(409, "The SQL of some champion features is missing")
        spec = from_yaml(spec_row.yaml)
        trained = {
            k: list(t.columns)
            for k, t in DataDescription.model_validate(version.source).tables.items()
        }
        source = connections.source(conn)
        try:
            return await asyncio.to_thread(
                self._score, run, folder, spec, graph, source, features, trained, cutoff, top_k
            )
        except scoring.SchemaMismatch as e:
            raise HTTPException(409, f"Schema changed since training: {e}") from None
        except (ConnectionFailure, EngineError) as e:
            raise HTTPException(502, f"Could not read the database: {e}") from None

    def _run_sql(self, source: Any, allowed: set[str], sql: str) -> pd.DataFrame:
        query = sql_guard.guard(sql, source.dialect, allowed_tables=allowed)
        table = sql_guard.execute(source, query, limit=MAX_ENTITIES, timeout_s=QUERY_TIMEOUT_S)
        return table.to_pandas()

    def _latest_cutoff(
        self, source: Any, allowed: set[str], graph: SchemaGraph, features: dict[str, str]
    ) -> datetime:
        """The start of the last day every table the features read has data up to."""
        ends = []
        text = " ".join(features.values())
        for t in graph.tables:
            if t.time_column and t.name in text:
                ref = f'"{t.db_schema}"."{t.name}"' if t.db_schema else f'"{t.name}"'
                frame = self._run_sql(
                    source, allowed, f'SELECT max("{t.time_column}") AS last FROM {ref}'
                )
                last = pd.to_datetime(frame["last"].iloc[0])
                if pd.notna(last):
                    ends.append(last)
        if not ends:
            raise HTTPException(
                422, "No table the features read has an event time to pick a cutoff from"
            )
        day = min(ends).floor("D")
        return datetime(day.year, day.month, day.day)  # noqa: DTZ001  (naive, like the data)

    def _score(
        self,
        run: Run,
        folder: Path,
        spec: Any,
        graph: SchemaGraph,
        source: Any,
        features: dict[str, str],
        trained: dict[str, list[str]],
        cutoff: datetime | None,
        top_k: int,
    ) -> RunScore:
        dialect = source.dialect
        if dialect not in ("postgres", "duckdb"):
            raise HTTPException(
                422,
                f"Scoring runs the features on the database itself, which needs Postgres or DuckDB, not {dialect}",
            )
        allowed = sql_guard.allowed_tables(source)
        used = {name: sql for name, sql in features.items()}
        entities_text = scoring.entities_sql(spec, graph, dialect, datetime(2000, 1, 1))  # noqa: DTZ001
        problems = scoring.required_columns_missing(
            trained, graph, {**used, "the task": entities_text}
        )
        if problems:
            raise scoring.SchemaMismatch("; ".join(problems))
        when = cutoff or self._latest_cutoff(source, allowed, graph, features)
        entities = scoring.entities_sql(spec, graph, dialect, when)
        ids = self._run_sql(source, allowed, entities)
        if ids.empty:
            raise HTTPException(422, f"No entity is eligible at {when:%Y-%m-%d}")
        ids["cutoff_time"] = pd.to_datetime(ids["cutoff_time"])
        keys = pd.MultiIndex.from_frame(ids[["entity_id", "cutoff_time"]])
        categories = json.loads((folder / CATEGORIES_FILE).read_text())
        columns: dict[str, Any] = {}
        for name, sql in features.items():
            got = self._run_sql(source, allowed, scoring.feature_sql(entities, sql, dialect))
            got["cutoff_time"] = pd.to_datetime(got["cutoff_time"])
            values = got.set_index(["entity_id", "cutoff_time"])["value"].reindex(keys)
            if name in categories:
                columns[name] = pd.Categorical(
                    values.astype("string").to_numpy(), categories=categories[name]
                )
            else:
                columns[name] = pd.to_numeric(values, errors="coerce").astype("float64").to_numpy()
        frame = pd.DataFrame(columns)
        booster = lgb.Booster(model_file=str(folder / MODEL_FILE))
        score, reasons = scoring.predict_with_reasons(booster, frame)
        rows = scoring.rank_rows(ids["entity_id"], score, reasons)
        stats = json.loads((folder / STATS_FILE).read_text())
        warnings = scoring.drift_warnings(
            frame, stats["features"], len(rows), stats["entities_per_cutoff"]
        )
        out = csv_path(folder, when)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        return RunScore(
            cutoff=when,
            summary=scoring.summary(rows, top_k),
            warnings=[{"name": w.name, "detail": w.detail} for w in warnings],
            preview=json.loads(rows.head(PREVIEW_ROWS).to_json(orient="records")),
            csv=f"/runs/{run.id}/scores/{when:%Y%m%dT%H%M%S}.csv",
        )

    def csv(self, project_id: str, run_id: str, stamp: str) -> Path:
        try:
            when = datetime.strptime(stamp, "%Y%m%dT%H%M%S")  # noqa: DTZ007
        except ValueError:
            raise HTTPException(404, "No such scores file") from None
        path = csv_path(artifact_dir(datasets.PROJECTS_DIR, project_id, run_id), when)
        if not path.exists():
            raise HTTPException(404, "These scores were not made; POST .../score first")
        return path
