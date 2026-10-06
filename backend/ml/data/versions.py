"""Immutable dataset versions named by the SHA-256 of their content.

A version is a read-only copy at ``<versions_dir>/<sha256>.<ext>``. Experiments
reference the version (its id, and the copy's path), never the file the user
uploaded, so editing or replacing that file can't change what a run trained on.
Identical bytes always map to the same version.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

CHUNK_BYTES = 1 << 20
SHORT_HASH_LEN = 12
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444


@dataclass(frozen=True)
class StoredVersion:
    id: str  # full SHA-256 hex digest of the content
    path: Path  # the read-only copy
    created: bool  # False when identical bytes were already stored

    @property
    def short_hash(self) -> str:
        return self.id[:SHORT_HASH_LEN]


def content_hash(path: Path) -> str:
    """SHA-256 of the file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_version(source: Path, versions_dir: Path) -> StoredVersion:
    """Copy ``source`` into ``versions_dir`` under its content hash, read-only.

    The copy is written to a temporary file in the same directory and renamed into
    place, so a reader never sees a half-written version.
    """
    source = Path(source)
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_id = content_hash(source)
    dest = versions_dir / f"{version_id}{source.suffix.lower()}"
    if dest.exists():
        return StoredVersion(id=version_id, path=dest, created=False)
    fd, tmp_name = tempfile.mkstemp(dir=versions_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
            shutil.copyfileobj(src, out)
        os.chmod(tmp_name, READ_ONLY)
        os.replace(tmp_name, dest)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    if content_hash(dest) != version_id:  # the source changed while it was being copied
        dest.chmod(stat.S_IWUSR | READ_ONLY)
        dest.unlink()
        raise RuntimeError(f"{source} changed while it was being versioned; try again")
    return StoredVersion(id=version_id, path=dest, created=True)


def version_id_of(path: str | Path, versions_dir: Path) -> str | None:
    """The version id when ``path`` is a stored version, else None."""
    p = Path(path).resolve()
    if p.parent != versions_dir.resolve():
        return None
    return p.stem if len(p.stem) == 64 else None
