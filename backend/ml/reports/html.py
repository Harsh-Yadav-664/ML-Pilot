"""The report as one self-contained HTML file: inline CSS, inline SVG, no script, no request."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from ml.reports.build import Chart, Report
from ml.reports.svg import gains_chart, split_chart

TEMPLATES = Path(__file__).parent / "templates"
CHARTS = {"split": split_chart, "gains": gains_chart}


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["chart"] = lambda c: Markup(CHARTS[c.kind](c.data))
    env.tests["chart_block"] = lambda b: isinstance(b, Chart)
    env.tests["string_block"] = lambda b: isinstance(b, str)
    env.tests["table_block"] = lambda b: hasattr(b, "headers")
    env.tests["code_block"] = lambda b: hasattr(b, "lang")
    return env


def to_html(report: Report) -> str:
    template = _env().get_template("report.html.j2")
    context: dict[str, Any] = {"report": report}
    return template.render(**context)
