"""CSV dataset loader (reads through the DuckDB engine, returns pandas at the model boundary)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.data.engine import arrow_to_pandas, read_file
from ml.data.ingestion.base import FileDatasetProvider


class CsvLoader(FileDatasetProvider):
    """Load CSV files into a pandas DataFrame."""

    SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt")

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        sep = kwargs.pop("sep", ",")
        if kwargs:
            raise TypeError(f"Unsupported CSV options: {sorted(kwargs)}")
        return arrow_to_pandas(read_file(path, sep=sep).read_all())
