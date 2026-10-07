"""The report as Markdown: the same sections as the HTML, with tables instead of charts."""

from __future__ import annotations

from ml.reports.build import Bullets, Chart, Code, Report, Table


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _fence(text: str) -> str:
    longest = max((len(line) - len(line.lstrip("`")) for line in text.splitlines()), default=0)
    return "`" * max(3, longest + 1)


def to_markdown(report: Report) -> str:
    out = [f"# {report.title}", "", " · ".join(report.meta), ""]
    for section in report.sections:
        out += [f"## {section.title}", ""]
        for block in section.blocks:
            if isinstance(block, str):
                out += [block, ""]
            elif isinstance(block, Table):
                out.append("| " + " | ".join(_cell(h) for h in block.headers) + " |")
                out.append("|" + "|".join(" --- " for _ in block.headers) + "|")
                out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in block.rows]
                out.append("")
            elif isinstance(block, Code):
                fence = _fence(block.text)
                out += [f"{fence}{block.lang}", block.text.rstrip("\n"), fence, ""]
            elif isinstance(block, Bullets):
                out += [f"- {' '.join(item.split())}" for item in block.items]
                out.append("")
            elif isinstance(block, Chart):
                continue  # the numbers of a chart are in the tables around it
    return "\n".join(out).rstrip("\n") + "\n"
