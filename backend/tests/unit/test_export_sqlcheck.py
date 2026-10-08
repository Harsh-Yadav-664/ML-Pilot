"""The guard rules copied into the export bundle (ml/export/templates/sqlcheck.py).

The bundle must not import MLPilot, so it carries a copy of the layer-1 rules of
ml/data/sql_guard.py. Every hostile statement of the main guard's matrix is refused here with
the same code, every allowed query passes, and the copied lists must equal the originals.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from ml.data import sql_guard
from ml.export import bundle

TEMPLATES = Path(bundle.__file__).parent / "templates"
FIXTURE = yaml.safe_load((Path(__file__).parents[1] / "fixtures" / "hostile_sql.yaml").read_text())

_spec = importlib.util.spec_from_file_location("bundle_sqlcheck", TEMPLATES / "sqlcheck.py")
assert _spec is not None and _spec.loader is not None
sqlcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sqlcheck)

HOSTILE = [
    (dialect, entry["sql"].replace("{t}", "orders"), entry["code"])
    for dialect in ("postgres", "duckdb")
    for entry in FIXTURE[dialect]
]
ALLOWED = [
    (dialect, sql.replace("{t}", "orders"))
    for dialect in ("postgres",)
    for sql in FIXTURE["allowed"][dialect]
]


@pytest.mark.parametrize(("dialect", "sql", "code"), HOSTILE)
def test_hostile_statements_are_refused_with_the_main_guards_code(
    dialect: str, sql: str, code: str
) -> None:
    with pytest.raises(sqlcheck.SqlRefused) as refused:
        sqlcheck.check(sql, dialect)
    assert refused.value.code == code


@pytest.mark.parametrize(("dialect", "sql"), ALLOWED)
def test_ordinary_queries_pass(dialect: str, sql: str) -> None:
    sqlcheck.check(sql, dialect)


def test_the_copied_rules_equal_the_main_guards() -> None:
    assert sqlcheck.DENIED_FUNCTIONS == sql_guard.DENIED_FUNCTIONS
    assert sqlcheck.DENIED_PREFIXES == sql_guard.DENIED_PREFIXES
    assert sqlcheck.DENIED_TABLES == sql_guard.DENIED_TABLES
    assert sqlcheck.FORBIDDEN_NODES == sql_guard.FORBIDDEN_NODES
    assert sqlcheck.ALLOWED_ROOTS == sql_guard.ALLOWED_ROOTS
