"""List the test files of one CI shard: ``python -m scripts.ci_shard <shards> <index>``.

The backend job splits ``pytest`` over several runners, each with its own Postgres service
container. Whole files stay together (their order and module fixtures are unchanged).
``PINNED`` puts the slowest files on chosen shards so the shards finish together; every other
file, including a new one, is dealt out round-robin, so no test file can fall out of CI.
``--check`` fails unless the shards together hold each file exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tests"

# file (relative to backend/) -> 1-based shard. Measured wall time per file, see ci.yml.
PINNED: dict[str, int] = {}


def test_files() -> list[str]:
    root = TESTS.parent
    return sorted(p.relative_to(root).as_posix() for p in TESTS.rglob("test_*.py"))


def assign(shards: int) -> dict[int, list[str]]:
    files = test_files()
    missing = set(PINNED) - set(files)
    if missing:
        raise SystemExit(f"pinned files that do not exist: {sorted(missing)}")
    out: dict[int, list[str]] = {i: [] for i in range(1, shards + 1)}
    unpinned: list[str] = []
    for f in files:
        pin = PINNED.get(f)
        if pin is None:
            unpinned.append(f)
        elif 1 <= pin <= shards:
            out[pin].append(f)
        else:
            raise SystemExit(f"{f} is pinned to shard {pin}, but there are {shards} shards")
    for n, f in enumerate(unpinned):
        out[n % shards + 1].append(f)
    return out


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[2] == "--check":
        shards = int(argv[1])
        flat = [f for files in assign(shards).values() for f in files]
        if sorted(flat) != test_files():
            print("shards do not hold each test file exactly once")
            return 1
        print(f"{len(flat)} test files over {shards} shards")
        return 0
    if len(argv) != 3:
        print(__doc__)
        return 2
    shards, index = int(argv[1]), int(argv[2])
    if not 1 <= index <= shards:
        print(f"index must be 1..{shards}")
        return 2
    print("\n".join(assign(shards)[index]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
