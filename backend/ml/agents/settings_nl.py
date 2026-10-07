"""Turn a sentence about a run's settings into a typed change (#59).

"budget $1", "at most 5 rounds", "ask me before each feature", "stop after 10 minutes".
The reading is by fixed patterns, not by a language model, so the same sentence always gives
the same change, and nothing is applied from a reading: the caller shows the change and asks.
What the patterns do not understand is returned as ``unrecognised``, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NUMBER = r"(\d+(?:\.\d+)?)"
UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hour": 3600,
    "hr": 3600,
}

SETTING_NAMES = {
    "max_cost_usd": "the language model budget (USD)",
    "max_seconds": "the time limit (seconds)",
    "max_proposals": "the number of proposals",
    "max_rounds": "the number of rounds",
    "patience": "patience (rounds without a gain before stopping)",
    "approval_mode": "who approves features",
}

APPROVAL_PATTERNS: list[tuple[str, str]] = [
    (r"(?:approve|check|confirm|review)\s+(?:each|every)\s+feature", "approve_each_feature"),
    (r"ask\s+me\s+(?:before|about)\s+(?:each|every)\s+feature", "approve_each_feature"),
    (r"(?:don'?t|do\s+not|never)\s+ask(?:\s+me)?", "auto"),
    (r"\b(?:run\s+on\s+)?auto(?:matic(?:ally)?|pilot)?\b", "auto"),
    (r"only\s+confirm\s+the\s+task", "confirm_task"),
]


@dataclass
class SettingsDiff:
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {"from", "to"}
    unrecognised: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.changes

    def summary(self) -> str:
        if not self.changes:
            return "No setting would change."
        lines = [
            f"{SETTING_NAMES.get(k, k)}: {v['from']!r} -> {v['to']!r}"
            for k, v in self.changes.items()
        ]
        return "; ".join(lines)


def _seconds(value: float, unit: str) -> float:
    unit = unit.lower().rstrip("s") or "s"
    return value * UNIT_SECONDS.get(unit, 1)


def parse(message: str, current: dict[str, Any]) -> SettingsDiff:
    """Read ``message`` against the ``current`` settings and return what would change."""
    text = message.strip()
    rest = text
    found: dict[str, Any] = {}

    def take(pattern: str, name: str, convert: Any = float) -> None:
        nonlocal rest
        m = re.search(pattern, rest, flags=re.IGNORECASE)
        if m:
            found[name] = convert(m)
            rest = rest[: m.start()] + " " + rest[m.end() :]

    take(
        rf"\$\s*{NUMBER}|{NUMBER}\s*(?:usd|dollars?)",
        "max_cost_usd",
        lambda m: float(m.group(1) or m.group(2)),
    )
    take(
        rf"{NUMBER}\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
        "max_seconds",
        lambda m: _seconds(float(m.group(1)), m.group(2)),
    )
    take(rf"patience\s*(?:of|to|=|:)?\s*{NUMBER}", "patience", lambda m: int(float(m.group(1))))
    take(
        rf"(?:stop\s+after\s+)?{NUMBER}\s+rounds?\s+without\s+(?:a\s+)?(?:gain|improvement|progress)",
        "patience",
        lambda m: int(float(m.group(1))),
    )
    take(
        rf"(?:max(?:imum)?|at\s+most|up\s+to|no\s+more\s+than|limit(?:\s+to)?)\s+(?:of\s+)?{NUMBER}\s+rounds?"
        rf"|(?:max(?:imum)?\s+)?rounds?\s*(?:of|to|=|:)\s*{NUMBER}",
        "max_rounds",
        lambda m: int(float(m.group(1) or m.group(2))),
    )
    take(
        rf"(?:max(?:imum)?|at\s+most|up\s+to|no\s+more\s+than|limit(?:\s+to)?)\s+(?:of\s+)?{NUMBER}\s+(?:proposals?|features?)"
        rf"|(?:max(?:imum)?\s+)?proposals?\s*(?:of|to|=|:)\s*{NUMBER}",
        "max_proposals",
        lambda m: int(float(m.group(1) or m.group(2))),
    )
    for pattern, mode in APPROVAL_PATTERNS:
        m = re.search(pattern, rest, flags=re.IGNORECASE)
        if m:
            found["approval_mode"] = mode
            rest = rest[: m.start()] + " " + rest[m.end() :]
            break

    diff = SettingsDiff()
    for name, value in found.items():
        if name in ("max_cost_usd", "max_seconds") and value <= 0:
            diff.unrecognised.append(f"{SETTING_NAMES[name]} must be above zero")
            continue
        if name in ("max_rounds", "max_proposals", "patience") and value < 1:
            diff.unrecognised.append(f"{SETTING_NAMES[name]} must be at least 1")
            continue
        before = current.get(name)
        if before != value:
            diff.changes[name] = {"from": before, "to": value}
    leftover = re.sub(r"[^a-z0-9]+", " ", rest.lower()).split()
    filler = {
        "set",
        "the",
        "to",
        "a",
        "an",
        "and",
        "please",
        "make",
        "it",
        "of",
        "for",
        "my",
        "budget",
        "limit",
        "time",
        "stop",
        "after",
        "at",
        "most",
        "max",
        "i",
        "want",
        "can",
        "you",
        "run",
        "with",
        "use",
        "only",
        "than",
        "more",
        "no",
        "up",
    }
    words = [w for w in leftover if w not in filler]
    if words and not found:
        diff.unrecognised.append(text)
    elif words:
        diff.unrecognised.append(" ".join(words))
    return diff
