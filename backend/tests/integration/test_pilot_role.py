"""The pilot kit's read-only role (#103): docs/pilot/readonly_role.sql run on the demo Postgres.

The script is run with psql as a privileged role (the compose service's `demo_admin` in CI;
MLPILOT_DEMO_PG_ADMIN_USER / _PASSWORD), then the new role is used through a plain connection
and through MLPilot's own source. The CI job `demo-db` sets MLPILOT_DEMO_PG_* and
MLPILOT_DEMO_PG_REQUIRED=1, which makes a missing database or psql a failure; everywhere else
this skips. The passwords here are fake values for the fake demo data.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pg8000.exceptions
import pg8000.native
import pytest

from ml.data.sources import ConnectionSpec, open_source

SCRIPT = Path(__file__).resolve().parents[3] / "docs" / "pilot" / "readonly_role.sql"
ROLE = "mlpilot_pilot_test"
DEFAULTS_ROLE = "mlpilot_pilot_test_defaults"
PASSWORD = "pilot-test-password-4417"  # fake: the demo database holds fake data
TABLES = "public.customers, public.orders, public.order_items, public.products"
GRANTED = {"customers", "orders", "order_items", "products"}


class Env:
    def __init__(self) -> None:
        self.host = os.environ["MLPILOT_DEMO_PG_HOST"]
        self.port = int(os.environ.get("MLPILOT_DEMO_PG_PORT", "5433"))
        self.database = os.environ.get("MLPILOT_DEMO_PG_DB", "demo")
        self.admin = os.environ["MLPILOT_DEMO_PG_ADMIN_USER"]
        self.admin_password = os.environ["MLPILOT_DEMO_PG_ADMIN_PASSWORD"]

    def run_script(self, **variables: str) -> subprocess.CompletedProcess[str]:
        """Run readonly_role.sql with psql as the privileged role; variables go in with -v."""
        args = [
            "psql", "-X", "-q", "-h", self.host, "-p", str(self.port),
            "-U", self.admin, "-d", self.database,
        ]  # fmt: skip
        for name, value in variables.items():
            args += ["-v", f"{name}={value}"]
        args += ["-f", str(SCRIPT)]
        env = {**os.environ, "PGPASSWORD": self.admin_password}
        return subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=120, check=False
        )

    def create(self, role: str = ROLE, **variables: str) -> subprocess.CompletedProcess[str]:
        variables = {
            "pilot_role": role,
            "pilot_password": PASSWORD,
            "pilot_tables": TABLES,
            **variables,
        }
        done = self.run_script(**variables)
        assert done.returncode == 0, done.stderr
        return done

    def drop(self, role: str = ROLE) -> None:
        done = self.run_script(pilot_role=role, revoke="drop")
        assert done.returncode == 0, done.stderr

    def admin_connection(self) -> pg8000.native.Connection:
        return pg8000.native.Connection(
            user=self.admin,
            password=self.admin_password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    def connect(self, role: str = ROLE) -> pg8000.native.Connection:
        return pg8000.native.Connection(
            user=role, password=PASSWORD, host=self.host, port=self.port, database=self.database
        )

    def spec(self, role: str = ROLE) -> ConnectionSpec:
        return ConnectionSpec(
            dialect="postgres",
            host=self.host,
            port=self.port,
            database=self.database,
            username=role,
            password=PASSWORD,
            ssl_mode="disable",
        )


@pytest.fixture(scope="module")
def env() -> Env:
    if not os.environ.get("MLPILOT_DEMO_PG_HOST"):
        if os.environ.get("MLPILOT_DEMO_PG_REQUIRED"):
            pytest.fail("MLPILOT_DEMO_PG_REQUIRED is set but the demo database is not configured")
        pytest.skip(
            "no demo database: start docker/demo-db/docker-compose.yml, set MLPILOT_DEMO_PG_*"
        )
    if not os.environ.get("MLPILOT_DEMO_PG_ADMIN_USER"):
        pytest.fail("MLPILOT_DEMO_PG_ADMIN_USER / _PASSWORD (a privileged role) are not set")
    if not shutil.which("psql"):
        if os.environ.get("MLPILOT_DEMO_PG_REQUIRED"):
            pytest.fail("psql is not installed")
        pytest.skip("psql is not installed")
    return Env()


@pytest.fixture
def pilot(env: Env) -> Iterator[Env]:
    """The role as the script creates it: 2 s statement timeout, 2 connections, four tables."""
    env.drop()
    env.create(pilot_statement_timeout="2s", pilot_connection_limit="2")
    try:
        yield env
    finally:
        env.drop()


def _sqlstate(error: pg8000.exceptions.DatabaseError) -> str:
    return str(error.args[0]["C"])


def _fails_with(code: str, conn: pg8000.native.Connection, sql: str) -> str:
    with pytest.raises(pg8000.exceptions.DatabaseError) as err:
        conn.run(sql)
    assert _sqlstate(err.value) == code, err.value
    return str(err.value.args[0]["M"])


@contextmanager
def _closing(conn: pg8000.native.Connection) -> Iterator[pg8000.native.Connection]:
    try:
        yield conn
    finally:
        conn.close()


def test_the_role_can_select_the_chosen_tables_and_only_those(pilot: Env) -> None:
    with _closing(pilot.connect()) as conn:
        counts = {t: conn.run(f"SELECT count(*) FROM {t}")[0][0] for t in sorted(GRANTED)}
        assert all(n > 0 for n in counts.values()), counts
        message = _fails_with("42501", conn, "SELECT count(*) FROM refunds")
    print(f"\nSELECT as {ROLE}: {counts}; refunds (not in the list): {message}")
    # MLPilot's own source, which runs everything through the SQL guard, sees the same four.
    source = open_source(pilot.spec())
    assert {t.name for t in source.list_tables()} == GRANTED
    n = source.query("SELECT count(*) AS n FROM customers", limit=1, timeout_s=10).column("n")
    assert n.to_pylist() == [counts["customers"]]


def test_the_role_cannot_write_with_the_default_session(pilot: Env) -> None:
    with _closing(pilot.connect()) as conn:
        assert conn.run("SHOW default_transaction_read_only") == [["on"]]
        insert = _fails_with("25006", conn, "INSERT INTO products VALUES (999999, 'x', 'y', 1)")
        update = _fails_with("25006", conn, "UPDATE customers SET country = 'XX'")
        create = _fails_with("25006", conn, "CREATE TABLE pilot_scratch (id int)")
    print(f"\nINSERT as {ROLE}: {insert}\nUPDATE: {update}\nCREATE TABLE: {create}")


def test_without_the_read_only_default_the_missing_privileges_still_stop_writes(
    pilot: Env,
) -> None:
    """The default is the second layer; the grants are the first. Switch the default off."""
    with _closing(pilot.connect()) as conn:
        conn.run("START TRANSACTION READ WRITE")
        insert = _fails_with("42501", conn, "INSERT INTO products VALUES (999999, 'x', 'y', 1)")
        conn.run("ROLLBACK")
        conn.run("SET default_transaction_read_only = off")
        delete = _fails_with("42501", conn, "DELETE FROM customers")
        create = _fails_with("42501", conn, "CREATE TABLE public.pilot_scratch (id int)")
    print(f"\nINSERT in a read-write transaction: {insert}\nDELETE: {delete}\nCREATE: {create}")


def test_the_statement_timeout_cancels_a_long_query(pilot: Env) -> None:
    with _closing(pilot.connect()) as conn:
        assert conn.run("SHOW statement_timeout") == [["2s"]]
        started = time.monotonic()
        message = _fails_with("57014", conn, "SELECT pg_sleep(30)")
        elapsed = time.monotonic() - started
    print(f"\nSELECT pg_sleep(30) stopped after {elapsed:.1f}s: {message}")
    assert 1.5 < elapsed < 10


def test_the_connection_limit_is_enforced(pilot: Env) -> None:
    with _closing(pilot.connect()) as first, _closing(pilot.connect()) as second:
        assert first.run("SELECT 1") == [[1]] and second.run("SELECT 1") == [[1]]
        with pytest.raises(pg8000.exceptions.DatabaseError) as err:
            pilot.connect()
    assert _sqlstate(err.value) == "53300", err.value  # too_many_connections
    print(f"\nthird connection: {err.value.args[0]['M']}")


def test_mlpilots_connection_test_calls_the_role_read_only(pilot: Env) -> None:
    privileges = open_source(pilot.spec()).privileges()
    print(f"\nMLPilot privilege check: can_write={privileges.can_write} notes={privileges.notes}")
    assert privileges.can_write is False and privileges.notes == []
    with _closing(pilot.admin_connection()) as admin:
        row = admin.run(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, rolcanlogin "
            "FROM pg_roles WHERE rolname = :r",
            r=ROLE,
        )
    assert row == [[False, False, False, False, False, True]]


def test_a_write_grant_added_later_is_noticed_and_the_session_default_still_holds(
    pilot: Env,
) -> None:
    """A control for the tests above: they would fail if the role could write."""
    with _closing(pilot.admin_connection()) as admin:
        admin.run(f"GRANT INSERT ON products TO {ROLE}")
    privileges = open_source(pilot.spec()).privileges()
    print(f"\nafter GRANT INSERT: can_write={privileges.can_write} notes={privileges.notes}")
    assert privileges.can_write is True and "write to 1 table" in privileges.notes[0]
    with _closing(pilot.connect()) as conn:
        _fails_with("25006", conn, "INSERT INTO products VALUES (999999, 'x', 'y', 1)")
        conn.run("START TRANSACTION READ WRITE")
        conn.run("INSERT INTO products VALUES (999999, 'x', 'y', 1)")  # the grant is the layer
        conn.run("ROLLBACK")


def test_the_documented_defaults_are_what_the_script_sets(env: Env) -> None:
    env.drop(DEFAULTS_ROLE)
    try:
        done = env.run_script(
            pilot_role=DEFAULTS_ROLE, pilot_password=PASSWORD, pilot_tables="public.customers"
        )
        assert done.returncode == 0, done.stderr
        with _closing(env.admin_connection()) as admin:
            ((limit, config),) = admin.run(
                "SELECT rolconnlimit, rolconfig FROM pg_roles WHERE rolname = :r", r=DEFAULTS_ROLE
            )
        print(f"\ndefaults: connection limit {limit}, settings {config}")
        assert limit == 4
        assert sorted(config) == [
            "default_transaction_read_only=on",
            "idle_in_transaction_session_timeout=60s",
            "statement_timeout=60s",
        ]
        with _closing(env.connect(DEFAULTS_ROLE)) as conn:
            assert conn.run("SELECT count(*) FROM customers")[0][0] > 0
            _fails_with("42501", conn, "SELECT count(*) FROM orders")
    finally:
        env.drop(DEFAULTS_ROLE)


def test_the_script_refuses_to_run_without_a_table_list(env: Env) -> None:
    env.drop()
    done = env.run_script(pilot_role=ROLE, pilot_password=PASSWORD)
    assert done.returncode != 0 and "pilot_tables is not set" in done.stderr
    with _closing(env.admin_connection()) as admin:
        assert admin.run("SELECT 1 FROM pg_roles WHERE rolname = :r", r=ROLE) == []
    error = next(line for line in done.stderr.splitlines() if "ERROR" in line)
    print(f"\nwithout pilot_tables: exit {done.returncode}, {error}")


def test_running_it_again_with_fewer_tables_takes_the_other_grants_away(pilot: Env) -> None:
    pilot.create(pilot_tables="public.customers")
    with _closing(pilot.connect()) as conn:
        assert conn.run("SELECT count(*) FROM customers")[0][0] > 0
        _fails_with("42501", conn, "SELECT count(*) FROM orders")
    assert {t.name for t in open_source(pilot.spec()).list_tables()} == {"customers"}


def test_lock_ends_open_sessions_and_logins_and_drop_removes_the_role(pilot: Env) -> None:
    open_session = pilot.connect()
    try:
        assert open_session.run("SELECT 1") == [[1]]
        done = pilot.run_script(pilot_role=ROLE, revoke="lock")
        assert done.returncode == 0, done.stderr
        with pytest.raises(
            (pg8000.exceptions.DatabaseError, pg8000.exceptions.InterfaceError, OSError)
        ):
            open_session.run("SELECT 1")  # the session was terminated
    finally:
        with contextlib.suppress(pg8000.exceptions.InterfaceError, OSError):  # the server ended it
            open_session.close()
    with pytest.raises(pg8000.exceptions.DatabaseError) as err:
        pilot.connect()
    print(f"\nlogin after revoke=lock: {err.value.args[0]['M']}")
    assert (
        _sqlstate(err.value) == "28000"
    )  # invalid_authorization_specification (not permitted to log in)

    pilot.create(pilot_statement_timeout="2s", pilot_connection_limit="2")  # run again: back
    with _closing(pilot.connect()) as conn:
        assert conn.run("SELECT 1") == [[1]]

    pilot.drop()
    with _closing(pilot.admin_connection()) as admin:
        assert admin.run("SELECT 1 FROM pg_roles WHERE rolname = :r", r=ROLE) == []
        assert (
            admin.run(
                "SELECT 1 FROM information_schema.role_table_grants WHERE grantee = :r", r=ROLE
            )
            == []
        )
    with pytest.raises(pg8000.exceptions.DatabaseError) as gone:
        pilot.connect()
    print(f"login after revoke=drop: {gone.value.args[0]['M']}")
