"""Keep the README's v1 checklist equal to the steps of the end-to-end test (issue #102).

    python scripts/works_today.py           rewrite the block in README.md
    python scripts/works_today.py --check   fail if the block differs from the test's step names

The block sits between the two marker comments; each line is one ``test.step('N. ...')`` of
frontend/e2e/v1.spec.ts, so the README cannot list a step the test does not run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "frontend" / "e2e" / "v1.spec.ts"
README = ROOT / "README.md"
BEGIN, END = "<!-- v1-steps:begin -->", "<!-- v1-steps:end -->"


def steps() -> list[str]:
    found = re.findall(r"test\.step\('(\d+)\. ([^']+)'", SPEC.read_text())
    if not found:
        raise SystemExit(f"no test.step found in {SPEC}")
    return [f"{n}. {text[0].upper()}{text[1:]}" for n, text in found]


def block() -> str:
    return "\n".join([BEGIN, *steps(), END])


def main(argv: list[str]) -> int:
    text = README.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md has no {BEGIN} ... {END} block")
    updated = pattern.sub(lambda _: block(), text)
    if "--check" in argv:
        if updated != text:
            print("README.md's v1 steps differ from frontend/e2e/v1.spec.ts; run scripts/works_today.py")
            return 1
        print("README.md's v1 steps match the end-to-end test")
        return 0
    README.write_text(updated)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
