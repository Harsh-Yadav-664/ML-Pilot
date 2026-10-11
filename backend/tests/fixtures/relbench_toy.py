"""A tiny RelBench dataset on disk, in RelBench's own format, for the harness tests (#65).

It is a miniature of rel-f1's ``driver-dnf``: drivers (no time), races and results (timed), and a
forecast task whose labels RelBench itself generates (``load_task(..., regenerate=True)`` runs the
task SQL) and which are then written as ``train/val/test.parquet``, like a hosted dataset. The
harness is then loaded through the real ``relbench.load_dataset`` and scored by the real
``task.evaluate``; only the data is synthetic. A driver's chance of not finishing is a fixed
latent number, so the history (and, through ``style``, a column of the driver row) carries signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TASK = "driver-dnf"
VAL_TIMESTAMP = "2002-07-01"
TEST_TIMESTAMP = "2003-01-01"
LABEL_SQL = """
SELECT t.timestamp AS date, r.driverId AS driverId, MAX(r.dnf)::INTEGER AS did_not_finish
FROM timestamps t, results r
WHERE r.date > t.timestamp AND r.date <= t.timestamp + INTERVAL '{timedelta}'
GROUP BY t.timestamp, r.driverId
"""


def build(root: Path, seed: int = 0) -> Path:
    """Write the dataset under ``root`` and return the path to load it from."""
    from relbench import load_dataset
    from relbench.manifest import DatasetManifest, TableSpec, TaskManifest

    rng = np.random.default_rng(seed)
    n_drivers = 80
    risk = rng.beta(2, 5, n_drivers)
    drivers = pd.DataFrame(
        {
            "driverId": np.arange(n_drivers),
            "nationality": rng.choice(["fr", "de", "it"], n_drivers),
            "style": risk + rng.normal(0, 0.1, n_drivers),
        }
    )
    race_dates = pd.date_range("2000-01-05", "2003-12-20", freq="9D")
    races = pd.DataFrame({"raceId": np.arange(len(race_dates)), "date": race_dates})
    rows = []
    for race in races.itertuples():
        for driver in rng.choice(n_drivers, 20, replace=False):
            rows.append((race.raceId, driver, race.date, int(rng.random() < risk[driver])))
    results = pd.DataFrame(rows, columns=["raceId", "driverId", "date", "dnf"])
    results.insert(0, "resultId", np.arange(len(results)))

    db = root / "db"
    db.mkdir(parents=True)
    for name, frame in (("drivers", drivers), ("races", races), ("results", results)):
        frame.to_parquet(db / f"{name}.parquet", index=False)
    DatasetManifest(
        name="toy-f1",
        val_timestamp=VAL_TIMESTAMP,
        test_timestamp=TEST_TIMESTAMP,
        tables={
            "drivers": TableSpec(pkey="driverId"),
            "races": TableSpec(pkey="raceId", time_col="date"),
            "results": TableSpec(
                pkey="resultId",
                time_col="date",
                fkeys={"raceId": "races", "driverId": "drivers"},
            ),
        },
    ).save(root / "manifest.yaml")
    task_dir = root / "tasks" / TASK
    TaskManifest(
        name=TASK,
        kind="forecast",
        task_type="binary_classification",
        entity_table="drivers",
        entity_col="driverId",
        target_col="did_not_finish",
        time_col="date",
        timedelta="30 days",
        num_eval_timestamps=3,
        sql=LABEL_SQL,
    ).save(task_dir / "manifest.yaml")
    # RelBench generates the labels from the task SQL; they are stored as a hosted dataset would
    generated = load_dataset(root).load_task(str(task_dir), regenerate=True)
    full = generated.get_db(upto_test_timestamp=False)
    for split in ("train", "val", "test"):
        table = generated.get_table(split, mask_input_cols=False, db=full)
        table.df.to_parquet(task_dir / f"{split}.parquet", index=False)
    return root
