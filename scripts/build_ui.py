"""Build the web UI into the Python package (issue #68).

    cd frontend && npm ci && cd ..
    python scripts/build_ui.py

The UI is built for the same origin (`/api/v1`) as one self-contained file and copied to
backend/mlpilot/ui/index.html, which FastAPI serves at `/` and the wheel ships as package data.
That folder is git-ignored: it is a build product, rebuilt for every wheel and Docker image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
TARGET = ROOT / "backend" / "mlpilot" / "ui"


def main() -> int:
    npm = shutil.which("npm")
    if npm is None:
        print("npm was not found: install Node.js 20.19 or newer to build the UI", file=sys.stderr)
        return 1
    if not (FRONTEND / "node_modules").is_dir():
        print("frontend/node_modules is missing: run `npm ci` in frontend/ first", file=sys.stderr)
        return 1
    # A development token (start.py writes frontend/.env.local) must never be baked into a build:
    # a variable set in the process wins over the .env files, so an empty one overrides it.
    env = {**os.environ, "VITE_API_BASE_URL": "/api/v1", "VITE_MLPILOT_TOKEN": ""}
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, env=env, check=True)
    built = FRONTEND / "dist" / "index.html"
    if not built.is_file():
        print(f"the build left no {built}", file=sys.stderr)
        return 1
    shutil.rmtree(TARGET, ignore_errors=True)
    TARGET.mkdir(parents=True)
    shutil.copy2(built, TARGET / "index.html")
    print(f"UI copied to {TARGET / 'index.html'} ({(TARGET / 'index.html').stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
