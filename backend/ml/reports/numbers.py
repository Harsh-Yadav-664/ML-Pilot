"""Every number in the report goes through ``Numbers.num``: it reads the value from the records by
its key, formats it, and writes down what it showed. Nothing types a figure into the text.

The log is the proof: a test finds every number in the Markdown and looks it up here, then checks
that the value is the one stored in the database.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# bare numerals: the sign, the unit and the words around them are written by the section builders
FORMATS: dict[str, Callable[[float], str]] = {
    "int": lambda v: f"{round(v)}",
    "f1": lambda v: f"{abs(v):.1f}",
    "f2": lambda v: f"{abs(v):.2f}",
    "f3": lambda v: f"{abs(v):.3f}",
    "f4": lambda v: f"{abs(v):.4f}",
    "pct1": lambda v: f"{abs(v) * 100:.1f}",
    "pct0": lambda v: f"{abs(v) * 100:.0f}",
    "usd": lambda v: f"{abs(v):.4f}",
    "secs": lambda v: f"{abs(v):.1f}",
}

_PATH = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)(?:\[(?P<idx>\d+)\]|\[(?P<k>[A-Za-z0-9_]+)=(?P<v>[^\]]+)\])?$"
)


class MissingRecord(KeyError):
    """The report asked for a stored value that is not there."""


def lookup(records: dict[str, Any], path: str) -> Any:
    """``final.test_metrics.pr_auc``, ``features[2].gain.mean_gain`` or ``schema.tables[key=orders].key``."""
    node: Any = records
    for part in path.split("."):
        m = _PATH.match(part)
        if m is None:
            raise MissingRecord(f"bad path {path!r}")
        if not isinstance(node, dict) or m["name"] not in node:
            raise MissingRecord(f"no record at {path!r} (stopped at {part!r})")
        node = node[m["name"]]
        if m["idx"] is not None:
            if not isinstance(node, list) or int(m["idx"]) >= len(node):
                raise MissingRecord(f"no item {m['idx']} at {path!r}")
            node = node[int(m["idx"])]
        elif m["k"] is not None:
            found = [x for x in node if isinstance(x, dict) and str(x.get(m["k"])) == m["v"]]
            if len(found) != 1:
                raise MissingRecord(f"{path!r}: {len(found)} items with {m['k']}={m['v']}")
            node = found[0]
    return node


@dataclass(frozen=True)
class Shown:
    key: str
    fmt: str
    value: float
    shown: str


class Numbers:
    """Formats numbers from the records and keeps the list of what was shown."""

    def __init__(self, records: dict[str, Any]) -> None:
        self.records = records
        self.shown: list[Shown] = []
        self.verbatim: list[str] = []

    def _bare(self, key: str, fmt: str) -> tuple[float, str]:
        value = lookup(self.records, key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MissingRecord(f"{key!r} is not a number: {value!r}")
        text = FORMATS[fmt](float(value))
        self.shown.append(Shown(key, fmt, float(value), text))
        return float(value), text

    def num(self, key: str, fmt: str) -> str:
        """The number; a negative value keeps its minus sign."""
        value, text = self._bare(key, fmt)
        return ("-" if value < 0 else "") + text

    def sign(self, key: str, fmt: str) -> str:
        """The number with an explicit sign, for gains."""
        value, text = self._bare(key, fmt)
        return ("-" if value < 0 else "+") + text

    def text(self, key: str) -> str:
        """A stored string shown as it is (ids, dates, names, stored messages)."""
        value = lookup(self.records, key)
        if value is None:
            raise MissingRecord(f"{key!r} is empty")
        text = str(value)
        self.verbatim.append(text)
        return text

    def cut(self, key: str) -> str:
        """'top 10%' from a stored metric name such as ``recall_at_10pct``: the cut-off is part of
        the name the run stored, so it is written down as stored text."""
        m = re.fullmatch(r"(\d+(?:_\d+)?)pct", key)
        if m is None:
            raise MissingRecord(f"{key!r} is not a cut-off name")
        text = f"top {m[1].replace('_', '.')}%"
        self.verbatim.append(text)
        return text

    def interval(self, key: str) -> str:
        """'95% interval' from the stored field name ``ci95``: the level is part of the name."""
        m = re.fullmatch(r"ci(\d+)", key)
        if m is None:
            raise MissingRecord(f"{key!r} is not an interval name")
        text = f"{m[1]}% interval"
        self.verbatim.append(text)
        return text

    def log(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.shown]


def find_numbers(text: str) -> list[str]:
    """The numerals in a text, not counting digits that are part of a word or identifier."""
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text)
