"""MLPilot's own data directories.

The API never accepts a file path from a client: data is addressed by its data version
id (the SHA-256 of the stored copy), which can only resolve inside VERSIONS_DIR.
"""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
DATASETS_DIR = BACKEND_DIR / "datasets"
# Read-only, content-addressed copies (ml/data/versions.py); experiments train on these.
VERSIONS_DIR = BACKEND_DIR / "data" / "versions"
