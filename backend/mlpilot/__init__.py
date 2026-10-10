"""The ``mlpilot`` package: the command line (``mlpilot.cli``), the migrations and the built UI.

MLPilot's code lives in the top-level packages ``app`` (API and services), ``ai`` (LLM gateway)
and ``ml`` (the engine); this thin package is what ``pip install mlpilot`` is named after and
what the ``mlpilot`` command starts. Keep this module free of imports: setuptools reads
``__version__`` from it, and the CLI sets its environment before anything else is imported.
"""

__version__ = "0.1.0"
