"""The evidence report of a run (#60): what was asked, what was done, what the numbers are.

``records`` (see ``records.py``) is everything the report may say, read from what the run stored.
``build_report`` turns it into sections; ``to_markdown`` and ``to_html`` write the same sections.
"""

from ml.reports.build import Report, build_report
from ml.reports.html import to_html
from ml.reports.markdown import to_markdown

__all__ = ["Report", "build_report", "to_html", "to_markdown"]
