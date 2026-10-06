"""Start MLPilot locally: the backend on 127.0.0.1:8000 and the UI on localhost:5173.

Every API call needs the local access token (backend/app/core/local_token.py). This
script creates it on first run and hands it to the UI through frontend/.env.local
(git-ignored), so local use stays one command.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core.local_token import load_or_create_token  # stdlib only

TOKEN_KEY = "VITE_MLPILOT_TOKEN"


def write_ui_token(token: str) -> None:
    """Put the token in frontend/.env.local, keeping any other lines already there."""
    env_file = ROOT / "frontend" / ".env.local"
    lines = []
    if env_file.exists():
        lines = [
            line
            for line in env_file.read_text().splitlines()
            if not line.startswith(f"{TOKEN_KEY}=")
        ]
    lines.append(f"{TOKEN_KEY}={token}")
    fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("=" * 50)
    print("🚀 Starting MLPilot (Backend + Frontend)")
    print("=" * 50)
    print("Press Ctrl+C to shut down both servers.")

    if not (ROOT / "backend").is_dir() or not (ROOT / "frontend").is_dir():
        print("Error: backend/ and frontend/ must sit next to start.py.")
        sys.exit(1)

    host = os.environ.get("MLPILOT_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"Warning: MLPILOT_HOST={host} makes the API reachable from other machines. "
            "Anyone with the token in ~/.mlpilot/token can use it."
        )
    write_ui_token(load_or_create_token())

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", host],
        cwd=ROOT / "backend",
        env={**os.environ, "MLPILOT_HOST": host},
    )
    # npm needs a shell on Windows; on Linux/macOS shell=True would drop "run dev".
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=ROOT / "frontend",
        shell=(os.name == "nt"),
    )
    children = [backend, frontend]

    def shutdown(code: int) -> None:
        print("\n" + "=" * 50)
        print("🛑 Shutting down MLPilot servers...")
        print("=" * 50)
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        sys.exit(code)

    signal.signal(signal.SIGTERM, lambda *_: shutdown(0))
    try:
        while True:
            for p in children:
                if p.poll() is not None:
                    print(f"A server exited with code {p.returncode}.")
                    shutdown(p.returncode or 1)
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(0)


if __name__ == "__main__":
    main()
