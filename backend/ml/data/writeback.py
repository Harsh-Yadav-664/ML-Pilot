"""Opt-in write-back of a run's scores to the user's Postgres (#172).

This is the only code that writes to a user database (AGENTS.md rule 2 makes this issue the
exception). It uses its own saved write-back connection, never the read-only one, and runs fixed
statements only: ``CREATE SCHEMA/TABLE IF NOT EXISTS`` and one ``INSERT``, all in
``mlpilot_predictions``. No SQL from a person or a model reaches it, so no other table can be
named, created or changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pg8000.exceptions

from ml.data.sources.base import ConnectionSpec
from ml.data.sources.postgres import PostgresSource

SCHEMA = "mlpilot_predictions"
TABLE = f"{SCHEMA}.scores"
TIMEOUT_S = 120
_STATEMENTS = (
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    (
        f"CREATE TABLE IF NOT EXISTS {TABLE} (run_id text NOT NULL, entity_id text NOT NULL, "
        "cutoff timestamptz NOT NULL, score double precision NOT NULL, "
        "created_at timestamptz NOT NULL DEFAULT now())"
    ),
)
_INSERT = (
    f"INSERT INTO {TABLE} (run_id, entity_id, cutoff, score) "
    "SELECT :run_id, unnest(CAST(:ids AS text[])), :cutoff, unnest(CAST(:scores AS float8[]))"
)


def write_scores(
    spec: ConnectionSpec,
    run_id: str,
    cutoff: datetime,
    entity_ids: Sequence[str],
    scores: Sequence[float],
) -> int:
    """Insert one row per scored entity, in one transaction; returns the number of rows."""
    if spec.dialect != "postgres":
        raise ValueError("Write-back is only available for Postgres")
    source = PostgresSource(spec)
    conn = source.connect(TIMEOUT_S)
    try:
        conn.run("BEGIN")
        conn.run(f"SET LOCAL statement_timeout = {TIMEOUT_S * 1000}")
        for statement in _STATEMENTS:
            conn.run(statement)
        conn.run(_INSERT, run_id=run_id, ids=list(entity_ids), cutoff=cutoff, scores=list(scores))
        conn.run("COMMIT")
    except (pg8000.exceptions.Error, OSError) as e:  # rolled back by closing without COMMIT
        raise source.failure(e) from None
    finally:
        conn.close()
    return len(entity_ids)
