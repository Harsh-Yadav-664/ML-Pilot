"""Build the export bundle of a finished run (#61).

Pure: everything comes in as ``BundleInput`` and goes out as ``{path: bytes}``, so the bundle
can be tested without a database. ``score.py`` and ``sqlcheck.py`` are copied from
``templates/`` unchanged; they import nothing from MLPilot.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
import yaml
from sqlglot import exp

from ml.data.schema_graph import SchemaGraph
from ml.tasks.labels import compile_labels
from ml.tasks.spec import TaskSpec

TEMPLATES = Path(__file__).parent / "templates"
DIALECTS = ("duckdb", "postgres")
CUTOFF_TOKEN = "__CUTOFF__"
FAR_FUTURE = datetime(2100, 1, 1)  # noqa: DTZ001  # keeps every cutoff of the spec: the export reads no data


class ExportError(ValueError):
    """The run cannot be exported, with the reason."""


@dataclass
class ExportFeature:
    name: str
    description: str
    sql: str  # DuckDB dialect, the contract of ml.features.compile: reads __labels


@dataclass
class BundleInput:
    run_id: str
    spec: TaskSpec
    graph: SchemaGraph
    dialect: str
    features: list[ExportFeature]
    model_text: str
    reference: pd.DataFrame  # entity_id, cutoff_time, label, score of the validation rows
    validation: dict[str, float]
    test: dict[str, float] | None
    manifest: dict[str, Any]
    versions: dict[str, str]
    categories: dict[str, list[str]] = field(default_factory=dict)  # text columns and levels
    report_html: str | None = None
    notes: list[str] = field(default_factory=list)


def transpile(sql: str, dialect: str) -> str:
    """A feature query (written for DuckDB) in the dialect of the database it will run on."""
    if dialect == "duckdb":
        return sql
    return ";".join(sqlglot.transpile(sql, read="duckdb", write=dialect, pretty=True))


def entity_query(spec: TaskSpec, graph: SchemaGraph, dialect: str) -> str:
    """Label-free query: the entities eligible at one cutoff, ``entity_id, cutoff_time``.

    It is the ``eligible`` step of the label query of the task, with the cutoffs replaced by
    the single ``__CUTOFF__`` placeholder, so what is scored is what was trained on.
    """
    tables = {t.key: t for t in graph.tables}

    def resolve(key: str) -> exp.Table:
        t = tables[key]
        return exp.table_(t.name, db=t.db_schema or None, quoted=True)

    compiled = compile_labels(spec, graph, dialect, resolve, FAR_FUTURE)
    ctes = {c.alias: c for c in compiled.select.ctes}
    if "eligible" not in ctes or "cutoffs" not in ctes:
        raise ExportError("the label query has no eligible step to reuse")
    cutoffs = sqlglot.parse_one(
        f"SELECT CAST('{CUTOFF_TOKEN}' AS TIMESTAMP) AS cutoff_time, "
        f"CAST('{CUTOFF_TOKEN}' AS TIMESTAMP) AS window_end",
        read="duckdb",
    )
    final = (
        exp.select("entity_id", "cutoff_time")
        .from_("eligible")
        .with_("cutoffs", as_=cutoffs)
        .with_("eligible", as_=ctes["eligible"].this)
    )
    return final.sql(dialect=dialect, pretty=True)


def entity_type(reference: pd.DataFrame) -> str:
    return "int" if pd.api.types.is_integer_dtype(reference["entity_id"]) else "str"


def _dbt_name(feature: str) -> str:
    return "f_" + re.sub(r"\W", "_", feature)


def _dbt_models(inp: BundleInput, entities: str, sqls: dict[str, str]) -> dict[str, str]:
    cutoff = "{{ var('cutoff') }}"
    out = {
        "dbt/dbt_project.yml": yaml.safe_dump(
            {
                "name": "mlpilot_features",
                "version": "1.0.0",
                "config-version": 2,
                "profile": "mlpilot_features",
                "model-paths": ["models"],
                "vars": {"cutoff": "2025-01-01"},
                "models": {"mlpilot_features": {"+materialized": "view"}},
            },
            sort_keys=False,
        ),
        "dbt/profiles.yml": (
            "# Example profile; point it at your warehouse (dbt-duckdb, dbt-postgres, ...).\n"
            "mlpilot_features:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n"
            "      path: ':memory:'\n"
        ),
        "dbt/models/entities.sql": entities.replace(CUTOFF_TOKEN, cutoff) + "\n",
    }
    joins, columns = [], []
    for i, f in enumerate(inp.features):
        model = _dbt_name(f.name)
        cte = "__labels AS (SELECT entity_id, cutoff_time FROM {{ ref('entities') }})"
        body = _with_labels(sqls[f.name], cte)
        out[f"dbt/models/features/{model}.sql"] = f"-- {f.description}\n{body}\n"
        columns.append(f'f{i}.value AS "{f.name}"')
        joins.append(
            f"LEFT JOIN {{{{ ref('{model}') }}}} f{i} ON f{i}.entity_id = e.entity_id "
            f"AND f{i}.cutoff_time = e.cutoff_time"
        )
    select = ",\n    ".join(["e.entity_id", "e.cutoff_time", *columns])
    out["dbt/models/features_all.sql"] = (
        f"SELECT\n    {select}\nFROM {{{{ ref('entities') }}}} e\n" + "\n".join(joins) + "\n"
    )
    return out


def _with_labels(sql: str, cte: str) -> str:
    match = re.match(r"\s*WITH\s+(RECURSIVE\s+)?", sql, flags=re.IGNORECASE)
    if match:
        return f"WITH {match.group(1) or ''}{cte}, {sql[match.end() :]}"
    return f"WITH {cte}\n{sql}"


def _readme(inp: BundleInput) -> str:
    v = inp.validation
    url = (
        "duckdb:///path/to.db" if inp.dialect == "duckdb" else "postgresql://user:password@host/db"
    )
    report = "| `report.html` | The evidence report of the run, readable offline |\n"
    test = ""
    if inp.test:
        test = (
            f"- Test PR-AUC {inp.test['pr_auc']:.4f} ({int(inp.test.get('n_test', 0))} rows), scored "
            "once at the end of the run by a model refit on train and validation. That refit model "
            "is not in this bundle: `model/model.txt` is the model the validation numbers belong to.\n"
        )
    notes = "".join(f"- {n}\n" for n in inp.notes)
    return f"""# {inp.spec.name}: exported from MLPilot run {inp.run_id}

{inp.spec.description or ""}

## What is in this folder

| File | What it is |
|---|---|
| `task.yaml` | The confirmed prediction task: entity, target, horizon, cutoffs, split |
| `manifest.json` | What is needed to reproduce the run: data version, seed, engine, budgets |
{report if inp.report_html else ""}| `features/*.sql` | One query per champion feature; reads `__labels(entity_id, cutoff_time)` |
| `entities.sql` | The entities eligible at a cutoff (`__CUTOFF__` is filled in by `score.py`) |
| `dbt/` | The same queries as a dbt project; `var('cutoff')` is the scoring date |
| `model/model.txt` | The LightGBM model in its native text format |
| `score.py`, `sqlcheck.py` | Score entities from your database, read-only; no MLPilot needed |
| `reference_validation.csv` | The validation rows with the scores MLPilot computed |

## Run it

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export MLPILOT_DB_URL='{url}'
python score.py --verify                       # recomputes the validation PR-AUC below
python score.py --cutoff 2025-01-01 --out scores.csv
```

The features are written for **{inp.dialect}**; table names are those of the task, with the
schema prefix the schema graph recorded.

## What the numbers mean

- Validation PR-AUC **{v["pr_auc"]:.4f}** against a base rate of {v["base_rate"]:.4f}
  ({int(v["n_val"])} validation rows). `score.py --verify` recomputes it from your database.
- Validation metrics are optimistic: the features and the stopping point were chosen on the
  same rows.
{test}{notes}"""


def build_bundle(inp: BundleInput) -> dict[str, bytes]:
    if inp.dialect not in DIALECTS:
        raise ExportError(f"dialect must be one of {DIALECTS}, not {inp.dialect!r}")
    if not inp.features:
        raise ExportError("the champion has no features to export")
    entities = entity_query(inp.spec, inp.graph, inp.dialect)
    config = {
        "dialect": inp.dialect,
        "run_id": inp.run_id,
        "entity_type": entity_type(inp.reference),
        "features": [f.name for f in inp.features],
        "categories": inp.categories,
        "validation": inp.validation,
    }
    text: dict[str, str] = {
        "README.md": _readme(inp),
        "task.yaml": yaml.safe_dump(inp.spec.model_dump(mode="json"), sort_keys=False),
        "manifest.json": json.dumps(
            {**inp.manifest, "versions": inp.versions}, indent=2, default=str
        ),
        "bundle.json": json.dumps(config, indent=2),
        "entities.sql": entities + "\n",
        "model/model.txt": inp.model_text,
        "requirements.txt": "".join(f"{k}=={v}\n" for k, v in sorted(inp.versions.items())),
        "reference_validation.csv": inp.reference.to_csv(index=False),
    }
    if inp.report_html:
        text["report.html"] = inp.report_html
    sqls = {f.name: transpile(f.sql, inp.dialect) for f in inp.features}
    for f in inp.features:
        text[f"features/{f.name}.sql"] = f"-- {f.description}\n{sqls[f.name]}\n"
    text.update(_dbt_models(inp, entities, sqls))
    files = {path: body.encode() for path, body in text.items()}
    for name in ("score.py", "sqlcheck.py"):
        files[name] = (TEMPLATES / name).read_bytes()
    return files


def zip_bundle(files: dict[str, bytes], root: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(f"{root}/{path}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[path])
    return buffer.getvalue()
