"""CSV dataset loader."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.data.ingestion.base import FileDatasetProvider


class CsvLoader(FileDatasetProvider):
    """Load CSV files into a pandas DataFrame."""

    SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt")

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        sep = kwargs.pop("sep", ",")
        return pd.read_csv(path, sep=sep, **kwargs)
