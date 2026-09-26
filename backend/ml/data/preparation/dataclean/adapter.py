"""DataCleanAdapter — stub for future DataClean project integration.

DataClean is a separate project that may be integrated as an optional
data preparation backend. This adapter bridges the MLPilot
DataPreparationProvider interface with the DataClean API.

IMPORTANT: This is intentionally NOT implemented in Phase 0.
Do not make architectural decisions based on DataClean internals.
Keep this file as a clean integration point.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ml.core.interfaces import (
    DataPreparationProvider,
    ProfileResult,
    ReadinessReport,
    LeakageWarning,
)


class DataCleanAdapter(DataPreparationProvider):
    """Adapter for the DataClean data preparation library.

    All methods raise NotImplementedError until DataClean is integrated.
    Replace the method bodies with DataClean API calls in Phase 1.
    """

    _NOT_IMPLEMENTED_MSG = (
        "DataClean integration is pending. "
        "Install and configure the DataClean package, "
        "then replace this stub with actual DataClean API calls."
    )

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def assess_readiness(self, df: pd.DataFrame, target_column: str) -> ReadinessReport:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def detect_leakage(self, df: pd.DataFrame, target_column: str) -> list[LeakageWarning]:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def prepare(self, df: pd.DataFrame, config: dict[str, Any]) -> tuple[Any, Any]:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def export_pipeline(self, pipeline: Any, path: str) -> str:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
