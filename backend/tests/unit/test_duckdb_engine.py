"""DuckDB as the single internal engine (#42): files become tables, reads go through DataSource."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.data.engine import (
    DuckDBSource,
    QueryRejected,
    QueryTimeout,
    RowLimitExceeded,
    TableRef,
    arrow_to_pandas,
    read_file,
    table_name_for,
)
from ml.data.ingestion.csv_loader import CsvLoader
from ml.data.ingestion.parquet_loader import ParquetLoader
from ml.data.profiling.profiler import DataProfiler
from ml.data.workspace import Workspace

TELECOM = Path(__file__).resolve().parents[2] / "datasets" / "telecom_churn.csv"

MESSY_CSV = (
    "id,plan,churn,tenure,total,flag\n"
    "1,a,Yes,5,10.5,True\n"
    "2,b,No,,20.0,False\n"
    "3,NA,Yes,7, ,True\n"
    "4,,No,8,30.25,False\n"
)


@pytest.fixture
def source() -> DuckDBSource:
    src = DuckDBSource()
    yield src
    src.close()


def test_a_csv_loads_with_the_same_types_and_nulls_as_pandas(tmp_path: Path) -> None:
    path = tmp_path / "messy.csv"
    path.write_text(MESSY_CSV)
    via_engine = CsvLoader().load(str(path))
    via_pandas = pd.read_csv(path)
    # Yes/No stays text (DuckDB's own inference would have made booleans of it) ...
    assert via_engine["churn"].tolist() == ["Yes", "No", "Yes", "No"]
    # ... integer NULLs become float NaN, a blank-space cell keeps the column text, as in pandas.
    assert str(via_engine["tenure"].dtype) == str(via_pandas["tenure"].dtype) == "float64"
    assert str(via_engine["total"].dtype) == str(via_pandas["total"].dtype)
    assert via_engine["plan"].isna().tolist() == via_pandas["plan"].isna().tolist()
    assert via_engine["tenure"].isna().tolist() == via_pandas["tenure"].isna().tolist()


def test_the_sample_dataset_loads_identically_to_pandas() -> None:
    via_engine = CsvLoader().load(str(TELECOM))
    via_pandas = pd.read_csv(TELECOM)
    assert list(via_engine.columns) == list(via_pandas.columns)
    pd.testing.assert_frame_equal(via_engine, via_pandas, check_dtype=False, check_exact=True)
    assert via_engine["Churn"].isin(["Yes", "No"]).all()


def test_parquet_goes_through_the_same_engine(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None], "c": [0.5, 1.5, 2.5]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    loaded = ParquetLoader().load(str(path))
    pd.testing.assert_frame_equal(loaded, df, check_dtype=False)


def test_unknown_file_types_and_options_are_rejected_loudly(tmp_path: Path) -> None:
    (tmp_path / "x.xlsx").write_bytes(b"x")
    with pytest.raises(Exception, match="Unsupported file type"):
        read_file(tmp_path / "x.xlsx")
    path = tmp_path / "a.csv"
    path.write_text("a\n1\n")
    with pytest.raises(TypeError):
        CsvLoader().load(str(path), nrows=1)


def test_table_names_come_from_file_names() -> None:
    assert table_name_for("Telecom Churn (2024).csv") == "telecom_churn_2024"
    assert table_name_for("2024.csv") == "t_2024"
    assert table_name_for("%%%.csv") == "data"


def test_data_source_lists_tables_and_describes_their_schema(
    source: DuckDBSource, tmp_path: Path
) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id,amount,note\n1,2.5,a\n2,3.5,b\n")
    ref = source.register_file(path, "orders")
    assert source.dialect == "duckdb"
    assert source.list_tables() == [TableRef("orders")]
    schema = source.table_schema(ref)
    assert [(c.name, c.type) for c in schema.columns] == [
        ("id", "BIGINT"),
        ("amount", "DOUBLE"),
        ("note", "VARCHAR"),
    ]
    assert source.row_count(ref) == 2


def test_query_returns_arrow_and_converts_to_pandas_only_at_the_boundary(
    source: DuckDBSource, tmp_path: Path
) -> None:
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,x\n2,y\n3,z\n")
    source.register_file(path, "t")
    table = source.query("SELECT a, b FROM t WHERE a >= ? ORDER BY a", [2], limit=10, timeout_s=5)
    assert type(table).__module__.startswith("pyarrow")
    assert arrow_to_pandas(table)["a"].tolist() == [2, 3]


def test_query_enforces_row_limit_timeout_and_select_only(
    source: DuckDBSource, tmp_path: Path
) -> None:
    path = tmp_path / "t.csv"
    path.write_text("a\n1\n2\n3\n")
    source.register_file(path, "t")
    with pytest.raises(RowLimitExceeded):
        source.query("SELECT * FROM t", limit=2, timeout_s=5)
    with pytest.raises(QueryTimeout):
        source.query(
            "SELECT count(*) FROM range(10000000000) a, range(1000000) b", limit=10, timeout_s=1
        )
    for bad in (
        "DROP TABLE t",
        "DELETE FROM t",
        "INSERT INTO t VALUES (9)",
        "SELECT 1; DROP TABLE t",
        "COPY t TO 'out.csv'",
        "ATTACH 'other.duckdb'",
    ):
        with pytest.raises(QueryRejected):
            source.query(bad, limit=10, timeout_s=5)
    assert source.row_count(TableRef("t")) == 3  # nothing was changed


def test_a_select_cannot_reach_files_outside_the_database(
    source: DuckDBSource, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    sql = f"SELECT * FROM read_csv('{secret}')"
    with pytest.raises(QueryRejected, match="read_csv"):  # the SQL guard refuses it ...
        source.query(sql, limit=10, timeout_s=5)
    with pytest.raises(duckdb.Error, match="(?i)permission|file system"):  # ... and so would DuckDB
        source._fetch(sql)


def test_fingerprint_changes_with_the_data_and_not_with_row_order(
    source: DuckDBSource, tmp_path: Path
) -> None:
    a = tmp_path / "a.csv"
    a.write_text("k,v\n1,x\n2,y\n")
    source.register_file(a, "t")
    first = source.fingerprint()
    assert source.fingerprint() == first
    a.write_text("k,v\n2,y\n1,x\n")  # same rows, other order
    source.register_file(a, "t")
    assert source.fingerprint() == first
    a.write_text("k,v\n1,x\n2,CHANGED\n")
    source.register_file(a, "t")
    assert source.fingerprint() != first


def test_a_workspace_registers_each_version_once_and_keeps_names_apart(tmp_path: Path) -> None:
    ws = Workspace(DuckDBSource(tmp_path / "p" / "work.duckdb"))
    first = tmp_path / "one.csv"
    first.write_text("a\n1\n2\n")
    second = tmp_path / "two.csv"
    second.write_text("a\n9\n")
    t1 = ws.register_version(first, "a" * 64, "sales.csv")
    assert ws.register_version(first, "a" * 64, "sales.csv") == t1  # idempotent
    t2 = ws.register_version(second, "b" * 64, "sales.csv")  # same name, other content
    assert (t1.name, t2.name) == ("sales", "sales_bbbbbbbb")
    assert ws.read_version("a" * 64).num_rows == 2
    assert ws.read_version("b" * 64).num_rows == 1
    ws.source.close()
    # The registry lives in the file: a new process finds the same tables.
    again = Workspace(DuckDBSource(tmp_path / "p" / "work.duckdb"))
    assert again.table_of("a" * 64) == t1
    assert [t.name for t in again.source.list_tables()] == ["sales", "sales_bbbbbbbb"]
    again.source.close()


def test_a_million_row_csv_loads_and_profiles_in_under_ten_seconds(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 1_000_000
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "age": rng.integers(18, 90, n),
            "tenure": rng.integers(0, 120, n),
            "monthly": rng.normal(60, 20, n).round(2),
            "total": rng.normal(2000, 900, n).round(2),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "region": rng.choice([f"r{i}" for i in range(20)], n),
            "paperless": rng.choice(["Yes", "No"], n),
            "support_calls": rng.integers(0, 12, n),
            "churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)  # building the file is not part of the timing

    ws = Workspace(DuckDBSource(tmp_path / "work.duckdb"))
    start = time.perf_counter()
    loaded = ws.load(path, "big.csv")
    loaded_at = time.perf_counter()
    profile = DataProfiler().profile(loaded, target_column="churn")
    end = time.perf_counter()
    ws.source.close()

    print(
        f"\n1M rows x 10 columns: load into DuckDB + read back {loaded_at - start:.2f}s, "
        f"profile {end - loaded_at:.2f}s, total {end - start:.2f}s"
    )
    assert profile.rows == n and profile.columns == 10
    assert end - start < 10
