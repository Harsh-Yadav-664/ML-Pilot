"""Where the built web UI is, when this install ships one.

``scripts/build_ui.py`` builds the React app into one self-contained file and copies it to
``mlpilot/ui/index.html``; the wheel and the Docker image carry it, a source checkout does not
(it runs the Vite dev server instead).
"""

from __future__ import annotations

from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[2] / "mlpilot" / "ui"


def ui_index() -> Path | None:
    index = UI_DIR / "index.html"
    return index if index.is_file() else None
