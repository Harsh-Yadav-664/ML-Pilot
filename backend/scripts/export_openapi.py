"""Write the API's OpenAPI schema to frontend/openapi.json (the frontend's contract).

python -m scripts.export_openapi          # rewrite the file
python -m scripts.export_openapi --check  # exit 1 if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.main import app

TARGET = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args()
    schema = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != schema:
            print(
                f"{TARGET} is out of date with the backend's response models.\n"
                "Run: cd backend && python -m scripts.export_openapi && "
                "cd ../frontend && npm run gen:api",
                file=sys.stderr,
            )
            return 1
        print(f"{TARGET.name} matches the backend")
        return 0
    TARGET.write_text(schema)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
