"""MLPilot's own data directories.

The API never accepts a file path from a client: data is addressed by its data version
id (the SHA-256 of the stored copy), which can only resolve inside VERSIONS_DIR.

From a source checkout the directories sit under ``backend/``. With ``MLPILOT_HOME`` set (the
``mlpilot`` command and the Docker image set it) they sit under that directory instead, because
an installed package must not write into ``site-packages``.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_HOME = "MLPILOT_HOME"


def home() -> Path:
    """Where MLPilot keeps its own files: ``$MLPILOT_HOME``, else ``backend/`` of a checkout."""
    explicit = os.environ.get(ENV_HOME, "").strip()
    return Path(explicit).expanduser().resolve() if explicit else BACKEND_DIR


_HOME = home()
UPLOAD_DIR = _HOME / "uploads"
# The bundled CSV samples are code, not state: they stay next to the source.
DATASETS_DIR = BACKEND_DIR / "datasets"
# Read-only, content-addressed copies (ml/data/versions.py); experiments train on these.
VERSIONS_DIR = _HOME / "data" / "versions"
# One DuckDB file per project: <PROJECTS_DIR>/<project id>/work.duckdb (ml/data/workspace.py).
PROJECTS_DIR = _HOME / "data" / "projects"
