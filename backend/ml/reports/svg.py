"""Two small inline SVG charts for the HTML report. No script, no external file.

The text in a chart is a name or a number the report already showed in its tables; the shapes
come from the stored values.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

WIDTH = 640


def _day(text: str) -> float:
    return datetime.fromisoformat(text).timestamp() / 86400


def split_chart(data: dict[str, Any]) -> str:
    """The cutoffs on a date line, coloured by period."""
    days = [_day(c) for c in data["cutoffs"]]
    val, test = _day(data["val_from"]), _day(data["test_from"])
    lo, hi = min([*days, val, test]), max([*days, val, test])
    span = (hi - lo) or 1.0

    def x(d: float) -> float:
        return 20 + (WIDTH - 40) * (d - lo) / span

    parts = [
        f'<svg viewBox="0 0 {WIDTH} 70" role="img" aria-label="Cutoffs by period" class="chart">',
        f'<rect x="20" y="18" width="{x(val) - 20:.1f}" height="18" class="train"/>',
        f'<rect x="{x(val):.1f}" y="18" width="{x(test) - x(val):.1f}" height="18" class="val"/>',
        f'<rect x="{x(test):.1f}" y="18" width="{WIDTH - 20 - x(test):.1f}" height="18" class="test"/>',
    ]
    for d, cutoff in zip(days, data["cutoffs"], strict=True):
        parts.append(
            f'<line x1="{x(d):.1f}" x2="{x(d):.1f}" y1="14" y2="40" class="tick"><title>{escape(cutoff)}</title></line>'
        )
    parts += [
        '<text x="20" y="60" class="lbl">train</text>',
        f'<text x="{x(val):.1f}" y="60" class="lbl">validation</text>',
        f'<text x="{x(test):.1f}" y="60" class="lbl">test</text>',
        "</svg>",
    ]
    return "".join(parts)


def gains_chart(data: dict[str, Any]) -> str:
    """Each tested feature's mean gain with its interval; the dashed line is its margin."""
    rows = data["rows"]
    lows = [r["lo"] for r in rows] + [0.0]
    highs = [r["hi"] for r in rows] + [r["margin"] for r in rows] + [0.0]
    lo, hi = min(lows), max(highs)
    span = (hi - lo) or 1.0
    left, right, row_h = 190, WIDTH - 70, 26

    def x(v: float) -> float:
        return left + (right - left) * (v - lo) / span

    height = 20 + row_h * len(rows)
    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="Paired gain per feature" class="chart">'
    ]
    parts.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="6" y2="{height - 8}" class="zero"/>')
    for i, r in enumerate(rows):
        y = 16 + row_h * i
        cls = "kept" if r["accepted"] else "not-kept"
        parts += [
            f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="lbl">{escape(r["name"])}</text>',
            f'<line x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{y}" y2="{y}" class="ci {cls}"/>',
            f'<circle cx="{x(r["mean"]):.1f}" cy="{y}" r="4" class="dot {cls}"/>',
            f'<line x1="{x(r["margin"]):.1f}" x2="{x(r["margin"]):.1f}" y1="{y - 8}" y2="{y + 8}" class="margin"/>',
            f'<text x="{right + 10}" y="{y + 4}" class="lbl">{escape(r["label"])}</text>',
        ]
    parts.append("</svg>")
    return "".join(parts)
