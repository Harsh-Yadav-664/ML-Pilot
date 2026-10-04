"""SQL dataset loader."""
from __future__ import annotations

from typing import Any
import pandas as pd
from sqlalchemy import create_engine
import json
import hashlib
import os

class SqlLoader:
    """Load data from a SQL database into a pandas DataFrame."""

    def load(self, connection_string: str, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute a read-only query and return a DataFrame.
        """
        # Ensure query is essentially read-only (basic check)
        lower_query = query.lower().strip()
        if not lower_query.startswith("select") and not lower_query.startswith("with"):
            raise ValueError("Only SELECT or WITH queries are permitted for direct data connection.")

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    def hash_connection(self, connection_string: str, query: str) -> str:
        """Create a deterministic hash for this query to use as a dataset ID."""
        raw = f"{connection_string}||{query}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
