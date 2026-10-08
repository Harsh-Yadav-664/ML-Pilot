"""Score entities with the exported MLPilot model, straight from your database.

    export MLPILOT_DB_URL='postgresql://user:password@host:5432/dbname'   # or duckdb:///path/to.db
    python score.py --cutoff 2025-01-01 --out scores.csv   # every eligible entity at that cutoff
    python score.py --verify                               # reproduce the validation metric

The connection is read-only, every statement passes sqlcheck.py first, and nothing here imports
MLPilot. The database URL is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not add the script's folder

import sqlcheck

CONFIG = json.loads((HERE / "bundle.json").read_text())
DIALECT: str = CONFIG["dialect"]
TIMEOUT_S = 600


def connect(url: str) -> Any:
    """A read-only connection."""
    parsed = urlparse(url)
    if DIALECT == "duckdb":
        import duckdb

        # duckdb:///relative.db or duckdb:////absolute/path.db
        return duckdb.connect(unquote(parsed.path)[1:], read_only=True)
    import pg8000.dbapi

    conn = pg8000.dbapi.connect(
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/"),
    )
    cur = conn.cursor()
    cur.execute("SET TRANSACTION READ ONLY")
    cur.execute(f"SET LOCAL statement_timeout = {TIMEOUT_S * 1000}")
    return conn


def fetch(conn: Any, sql: str) -> pd.DataFrame:
    sqlcheck.check(sql, DIALECT)
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columns)


def labels_cte(entities: pd.DataFrame) -> str:
    """The ``__labels`` relation for given (entity_id, cutoff_time) rows, as a VALUES list."""
    kind = CONFIG["entity_type"]
    rows = []
    for entity, cutoff in zip(entities["entity_id"], entities["cutoff_time"], strict=True):
        key = str(int(entity)) if kind == "int" else "'" + str(entity).replace("'", "''") + "'"
        stamp = pd.Timestamp(cutoff).strftime("%Y-%m-%d %H:%M:%S.%f")
        rows.append(f"({key}, TIMESTAMP '{stamp}')")
    return f"__labels(entity_id, cutoff_time) AS (SELECT * FROM (VALUES {', '.join(rows)}) AS v)"


def with_labels(feature_sql: str, cte: str) -> str:
    feature_sql = "\n".join(
        line for line in feature_sql.splitlines() if not line.lstrip().startswith("--")
    )
    match = re.match(r"\s*WITH\s+(RECURSIVE\s+)?", feature_sql, flags=re.IGNORECASE)
    if match:
        return f"WITH {match.group(1) or ''}{cte}, {feature_sql[match.end() :]}"
    return f"WITH {cte} {feature_sql}"


def eligible(conn: Any, cutoff: str) -> pd.DataFrame:
    stamp = datetime.fromisoformat(cutoff).strftime("%Y-%m-%d %H:%M:%S")  # also validates it
    sql = (HERE / "entities.sql").read_text().replace("__CUTOFF__", stamp)
    frame = fetch(conn, sql)
    frame["cutoff_time"] = pd.to_datetime(frame["cutoff_time"])
    return frame


def feature_frame(conn: Any, entities: pd.DataFrame) -> pd.DataFrame:
    cte = labels_cte(entities)
    out = entities[["entity_id", "cutoff_time"]].copy()
    out["cutoff_time"] = pd.to_datetime(out["cutoff_time"])
    keys = pd.MultiIndex.from_frame(out)
    columns = {}
    for feature in CONFIG["features"]:
        sql = with_labels((HERE / "features" / f"{feature}.sql").read_text(), cte)
        got = fetch(conn, sql)
        got["cutoff_time"] = pd.to_datetime(got["cutoff_time"])
        values = got.set_index(["entity_id", "cutoff_time"])["value"].reindex(keys)
        if feature in CONFIG["categories"]:  # a text attribute: the levels the model was trained on
            levels = CONFIG["categories"][feature]
            columns[feature] = pd.Categorical(values.astype("string").to_numpy(), categories=levels)
        else:
            columns[feature] = pd.to_numeric(values, errors="coerce").astype("float64").to_numpy()
    return pd.concat([out, pd.DataFrame(columns, index=out.index)], axis=1)


def predict(frame: pd.DataFrame) -> pd.Series:
    booster = lgb.Booster(model_file=str(HERE / "model" / "model.txt"))
    names = booster.feature_name()
    return pd.Series(booster.predict(frame[names]), index=frame.index)


def verify(conn: Any) -> int:
    reference = pd.read_csv(HERE / "reference_validation.csv", parse_dates=["cutoff_time"])
    frame = feature_frame(conn, reference)
    score = predict(frame)
    got = float(average_precision_score(reference["label"], score))
    want = float(CONFIG["validation"]["pr_auc"])
    worst = float(np.abs(score.to_numpy() - reference["score"].to_numpy()).max())
    print(f"validation PR-AUC reported {want:.9f}, recomputed from your database {got:.9f}")
    print(f"largest difference in a single score: {worst:.3e}  (rows: {len(reference)})")
    ok = abs(got - want) <= 1e-6
    print("OK: reproduced within 1e-6" if ok else "NOT reproduced: the data differs from the run")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cutoff", help="score every eligible entity at this date or timestamp")
    parser.add_argument("--out", default="scores.csv")
    parser.add_argument("--verify", action="store_true", help="reproduce the validation metric")
    args = parser.parse_args()
    url = os.environ.get("MLPILOT_DB_URL")
    if not url:
        print("Set MLPILOT_DB_URL to the database URL (see the top of score.py).", file=sys.stderr)
        return 2
    if not args.verify and not args.cutoff:
        parser.error("give --cutoff or --verify")
    conn = connect(url)
    try:
        if args.verify:
            return verify(conn)
        entities = eligible(conn, args.cutoff)
        frame = feature_frame(conn, entities)
        result = frame[["entity_id", "cutoff_time"]].assign(score=predict(frame))
        result.sort_values("score", ascending=False).to_csv(args.out, index=False)
        print(f"wrote {len(result)} scores to {args.out}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
