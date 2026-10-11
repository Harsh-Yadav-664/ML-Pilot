"""The SQL guard and the MySQL connector (#174) against a real MySQL (the CI service container).

Skipped unless MLPILOT_TEST_MYSQL_* is set. The same contract as Postgres
(test_sql_guard_postgres.py, test_live_sources.py):

* introspection (tables, columns, primary and foreign keys), queries, row limit, timeout;
* Layer 1: every hostile query in tests/fixtures/hostile_sql.yaml is refused, with its code;
* Layer 2: with the parser switched off, the read-only session still refuses writes;
* Layer 3: the privilege check reports ``can_write`` for a user that may INSERT.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymysql
import pytest
import yaml

from ml.data import sql_guard
from ml.data.engine import QueryRejected, QueryTimeout, RowLimitExceeded, TableRef
from ml.data.sources import ConnectionFailure, ConnectionSpec, open_source
from ml.data.sql_guard import SqlRejected
from tests.fixtures.mysql import mysql_spec

BACKEND = Path(__file__).resolve().parents[2]
HOSTILE = yaml.safe_load((BACKEND / "tests" / "fixtures" / "hostile_sql.yaml").read_text())
TABLE = "mlpilot_guard_orders"
INSERTER, INSERTER_PASSWORD = "mlpilot_guard_inserter", "inserter-pass-5512"
READER, READER_PASSWORD = "mlpilot_guard_reader", "reader-pass-8820"
PWNED = Path("/tmp/mlpilot_guard_pwned")


def _admin(spec: ConnectionSpec) -> Any:
    return pymysql.connect(
        user=spec.username or "",
        password=spec.password or "",
        host=spec.host or "",
        port=spec.port or 3306,
        database=spec.database,
        autocommit=True,
    )


def _run(admin: Any, sql: str) -> list[tuple[Any, ...]]:
    with admin.cursor() as cur:
        cur.execute(sql)
        return [tuple(r) for r in cur.fetchall()]


@pytest.fixture
def my() -> Iterator[ConnectionSpec]:
    """Tables with a foreign key, a user that may INSERT and a user that may only SELECT."""
    spec = mysql_spec()
    admin = _admin(spec)
    db = spec.database
    for sql in (
        "SET FOREIGN_KEY_CHECKS = 0",
        f"DROP TABLE IF EXISTS {TABLE}, mlpilot_guard_new, mlpilot_guard_customers",
        "SET FOREIGN_KEY_CHECKS = 1",
        "CREATE TABLE mlpilot_guard_customers (id int PRIMARY KEY, name text)",
        (
            f"CREATE TABLE {TABLE} (id int PRIMARY KEY, customer int, amount decimal(10,2), "
            "FOREIGN KEY (customer) REFERENCES mlpilot_guard_customers (id))"
        ),
        "INSERT INTO mlpilot_guard_customers VALUES (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')",
        (
            f"INSERT INTO {TABLE} WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n "
            "WHERE i < 40) SELECT i, i % 4, i * 2.5 FROM n"
        ),
    ):
        _run(admin, sql)
    for user, password, grant in (
        (INSERTER, INSERTER_PASSWORD, "SELECT, INSERT"),
        (READER, READER_PASSWORD, "SELECT"),
    ):
        _run(admin, f"DROP USER IF EXISTS '{user}'@'%'")
        _run(admin, f"CREATE USER '{user}'@'%' IDENTIFIED BY '{password}'")
        _run(admin, f"GRANT {grant} ON {db}.* TO '{user}'@'%'")
    PWNED.unlink(missing_ok=True)
    try:
        yield spec
    finally:
        _run(admin, "SET FOREIGN_KEY_CHECKS = 0")
        _run(admin, f"DROP TABLE IF EXISTS {TABLE}, mlpilot_guard_new, mlpilot_guard_customers")
        for user in (INSERTER, READER):
            _run(admin, f"DROP USER IF EXISTS '{user}'@'%'")
        admin.close()


def _state(spec: ConnectionSpec) -> tuple[Any, ...]:
    """Everything a write could change: the rows, the table list, the columns."""
    admin = _admin(spec)
    try:
        return (
            _run(admin, f"SELECT id, customer, amount FROM {TABLE} ORDER BY id"),
            _run(
                admin,
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name LIKE 'mlpilot_guard%' ORDER BY 1",
            ),
            _run(
                admin,
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema = DATABASE() AND table_name = '{TABLE}' "
                "ORDER BY ordinal_position",
            ),
        )
    finally:
        admin.close()


def test_mysql_lists_describes_and_queries_tables(my: ConnectionSpec) -> None:
    src = open_source(my)
    src.check()
    assert src.server_version()[0].isdigit()
    table = TableRef(TABLE, my.database)
    assert table in src.list_tables()
    schema = src.table_schema(table)
    assert [(c.name, c.type) for c in schema.columns] == [
        ("id", "int"),
        ("customer", "int"),
        ("amount", "decimal"),
    ]
    assert schema.primary_key == ["id"]
    (fk,) = schema.foreign_keys
    assert (fk.columns, fk.ref_table, fk.ref_columns) == (
        ["customer"],
        TableRef("mlpilot_guard_customers", my.database),
        ["id"],
    )
    result = src.query(
        f"SELECT customer, count(*) AS n FROM {TABLE} GROUP BY customer ORDER BY 1",
        limit=100,
        timeout_s=10,
    )
    assert result.num_rows == 4 and set(result.column("n").to_pylist()) == {10}
    assert len(src.fingerprint()) == 64


def test_mysql_enforces_select_only_row_limit_and_timeout(my: ConnectionSpec) -> None:
    src = open_source(my)
    for bad in (f"DROP TABLE {TABLE}", f"DELETE FROM {TABLE}", f"SELECT 1; DROP TABLE {TABLE}"):
        with pytest.raises(QueryRejected):
            src.query(bad, limit=10, timeout_s=5)
    with pytest.raises(RowLimitExceeded):
        src.query(f"SELECT * FROM {TABLE}", limit=5, timeout_s=5)
    slow = (  # 40^6 rows, each hashed: MySQL cannot shortcut it the way it does a bare count(*)
        f"SELECT count(*) FROM {TABLE} a, {TABLE} b, {TABLE} c, {TABLE} d, {TABLE} e, {TABLE} f "
        "WHERE MD5(a.id + b.id + c.id + d.id + e.id + f.id) LIKE 'zzz%'"
    )
    with pytest.raises(QueryTimeout):
        src.query(slow, limit=5, timeout_s=1)
    n = src.query(f"SELECT count(*) AS n FROM {TABLE}", limit=1, timeout_s=5).column("n")
    assert n.to_pylist() == [40]  # nothing was dropped or deleted


# Single statements a parser bug could let through. Each must fail in the session itself.
WRITES_WITHOUT_LAYER_1 = [
    f"INSERT INTO {TABLE} VALUES (99, 1, 1)",
    f"UPDATE {TABLE} SET amount = 0",
    f"DELETE FROM {TABLE}",
    f"REPLACE INTO {TABLE} VALUES (1, 1, 1)",
    f"TRUNCATE TABLE {TABLE}",
    f"DROP TABLE {TABLE}",
    "CREATE TABLE mlpilot_guard_new (a int)",
    f"ALTER TABLE {TABLE} ADD COLUMN extra int",
    f"CREATE TABLE mlpilot_guard_new AS SELECT * FROM {TABLE}",
]


def _unguarded(src: Any, sql: str) -> Any:
    """Layer 1 switched off: the SQL goes straight into the read-only session."""
    with sql_guard.read_only_session(src, 5) as session:
        return session.fetch_rows(sql, [], max_rows=None, max_bytes=10**6)


@pytest.mark.parametrize("sql", WRITES_WITHOUT_LAYER_1)
def test_with_the_parser_off_mysql_still_refuses_writes(my: ConnectionSpec, sql: str) -> None:
    src = open_source(my)
    before = _state(my)
    with pytest.raises(ConnectionFailure) as err:
        _unguarded(src, sql)
    # 1792 ER_CANT_EXECUTE_IN_READ_ONLY_TRANSACTION
    assert re.search(r"error code 1792", str(err.value)), err.value
    assert _state(my) == before


def test_with_the_parser_off_mysql_refuses_a_second_statement(my: ConnectionSpec) -> None:
    """PyMySQL does not enable multi-statements, so the server refuses a smuggled second one."""
    src = open_source(my)
    before = _state(my)
    for sql in (f"COMMIT; DELETE FROM {TABLE}", f"SELECT 1; DROP TABLE {TABLE}"):
        with pytest.raises(ConnectionFailure) as err:
            _unguarded(src, sql)
        assert "error code 1064" in str(err.value), err.value  # ER_PARSE_ERROR
    assert _state(my) == before


def test_the_session_is_read_only_with_a_timeout_for_every_connection(my: ConnectionSpec) -> None:
    src = open_source(my)
    _, rows = _unguarded(src, "SELECT @@transaction_read_only, @@max_execution_time")
    assert rows == [(1, 5000)]


def test_the_hostile_matrix_is_refused_by_mysql(my: ConnectionSpec) -> None:
    src = open_source(my)
    before = _state(my)
    for case in HOSTILE["mysql"]:
        sql = case["sql"].replace("{t}", TABLE)
        with pytest.raises(SqlRejected) as err:
            src.query(sql, limit=100, timeout_s=5)
        assert err.value.code == case["code"], (sql, err.value.reason)
    for sql in HOSTILE["allowed"]["mysql"]:
        assert src.query(sql.replace("{t}", TABLE), limit=1000, timeout_s=10).num_rows > 0
    assert _state(my) == before
    assert not PWNED.exists()
    print(f"\n{len(HOSTILE['mysql'])} hostile queries refused by the guard against MySQL")


def test_the_privilege_check_tells_a_writer_from_a_reader(my: ConnectionSpec) -> None:
    writer = open_source(
        dataclasses.replace(my, username=INSERTER, password=INSERTER_PASSWORD)
    ).privileges()
    assert writer.can_write and any("INSERT" in n for n in writer.notes), writer.notes
    reader = open_source(dataclasses.replace(my, username=READER, password=READER_PASSWORD))
    privileges = reader.privileges()
    assert not privileges.can_write, privileges.notes
    n = reader.query(f"SELECT count(*) AS n FROM {TABLE}", limit=1, timeout_s=5).column("n")
    assert n.to_pylist() == [40]
    print(f"\ninserter: can_write={writer.can_write} notes={writer.notes}")
    print(f"reader:   can_write={privileges.can_write} notes={privileges.notes}")


def test_mysql_errors_are_short_fixed_messages(my: ConnectionSpec) -> None:
    cases = {
        "auth_failed": dataclasses.replace(my, password="definitely-wrong-pw"),
        "database_not_found": dataclasses.replace(my, database="no_such_database_here"),
        "host_unreachable": dataclasses.replace(my, port=1),
    }
    for code, spec in cases.items():
        with pytest.raises(ConnectionFailure) as err:
            open_source(spec).check()
        assert err.value.code == code, (code, err.value.message)
        for secret in (spec.password or "", spec.host or "", spec.username or ""):
            assert secret not in err.value.message
    with pytest.raises(ConnectionFailure) as err:
        open_source(dataclasses.replace(my, username=READER, password=READER_PASSWORD)).query(
            "SELECT * FROM mlpilot_guard_not_there", limit=1, timeout_s=5
        )
    assert "1146" in str(err.value) and READER not in str(err.value)  # ER_NO_SUCH_TABLE


def test_a_mysql_source_does_not_show_its_password_in_its_repr(my: ConnectionSpec) -> None:
    assert (my.password or "") not in repr(my) and (my.password or "") not in repr(open_source(my))
