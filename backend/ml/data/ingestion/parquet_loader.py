"""Parquet dataset loader."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ml.data.ingestion.base import FileDatasetProvider


class ParquetLoader(FileDatasetProvider):
    """Load Parquet files into a pandas DataFrame."""

    SUPPORTED_EXTENSIONS = (".parquet",)

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)
