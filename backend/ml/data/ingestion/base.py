"""Base DatasetProvider for file-based ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.core.interfaces import DatasetProvider


class FileDatasetProvider(DatasetProvider):
    """Base class for file-based dataset loaders."""

    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def schema(self, df: pd.DataFrame) -> dict[str, str]:
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError
