"""Parquet dataset loader (reads through the DuckDB engine)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.data.engine import arrow_to_pandas, read_file
from ml.data.ingestion.base import FileDatasetProvider


class ParquetLoader(FileDatasetProvider):
    """Load Parquet files into a pandas DataFrame."""

    SUPPORTED_EXTENSIONS = (".parquet",)

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        if kwargs:
            raise TypeError(f"Unsupported Parquet options: {sorted(kwargs)}")
        return arrow_to_pandas(read_file(path).read_all())
