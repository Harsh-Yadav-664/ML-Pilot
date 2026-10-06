"""Only files inside MLPilot's own data directories may be loaded by the API."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
DATASETS_DIR = BACKEND_DIR / "datasets"
# Read-only, content-addressed copies (ml/data/versions.py); experiments train on these.
VERSIONS_DIR = BACKEND_DIR / "data" / "versions"

# Tests may replace this list with their own temporary directories.
ALLOWED_DATA_DIRS: list[Path] = [UPLOAD_DIR, DATASETS_DIR]


def safe_dataset_path(path: str) -> str:
    """Resolve a client-supplied dataset path and reject anything outside the data dirs.

    Relative paths are resolved against the backend directory. Returns the
    absolute path; raises HTTP 400 for anything else (traversal, other files,
    non-CSV files).
    """
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else BACKEND_DIR / raw).resolve()
    allowed = any(candidate.is_relative_to(d.resolve()) for d in [*ALLOWED_DATA_DIRS, VERSIONS_DIR])
    if not allowed or candidate.suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="dataset_path must be a CSV uploaded to MLPilot or a bundled sample dataset",
        )
    return str(candidate)
