"""The README's command-line steps, as a shell script (issue #68).

    python scripts/readme_cli.py            print the commands, one per line
    python scripts/readme_cli.py --check    fail unless the block is the five documented commands

The block sits between the two marker comments in README.md. CI's `pip-install` job runs exactly
what this prints, in a clean virtualenv that has only the installed wheel, so the README cannot
list a command that the job does not run (backend/tests/unit/test_cli.py also parses every
line with the real command line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
BEGIN, END = "<!-- cli-steps:begin -->", "<!-- cli-steps:end -->"
# The five steps, in order; setup lines (export ...) may come before them.
STEPS = ("mlpilot connect", "mlpilot task draft", "mlpilot task confirm", "mlpilot run", "mlpilot report")


def commands(text: str | None = None) -> list[str]:
    text = README.read_text(encoding="utf-8") if text is None else text
    found = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, re.S)
    if not found:
        raise SystemExit(f"README.md has no {BEGIN} ... {END} block")
    lines = [
        line.strip()
        for line in found.group(1).splitlines()
        if line.strip() and not line.strip().startswith(("```", "#"))
    ]
    if not lines:
        raise SystemExit("the README's command block is empty")
    return lines


def subcommand(line: str) -> str:
    """``mlpilot task draft --question ...`` -> ``mlpilot task draft``."""
    words: list[str] = []
    for word in line.split():
        if word.startswith("-"):
            break
        words.append(word)
    return " ".join(words)


def main(argv: list[str]) -> int:
    lines = commands()
    if "--check" in argv:
        found = [subcommand(line) for line in lines if line.startswith("mlpilot ")]
        if found != list(STEPS):
            print(f"README.md's block must run exactly: {', '.join(STEPS)}\nfound: {found}")
            return 1
        if any("\\" in line for line in lines):
            print("README.md's block must not use line continuations (one command per line)")
            return 1
        print(f"README.md's block has the {len(STEPS)} documented commands")
        return 0
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
