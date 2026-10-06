"""Every call to the LLM gateway in app/ and ml/ sends a prompt built by the context builder (#48).

At run time the gateway raises for anything else; this test checks the source so that a new call
site cannot be added with a string and only fail on the day it runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
GATEWAY_METHODS = {
    "complete",
    "complete_structured",
    "complete_result",
    "complete_structured_result",
}


def builds_prompt(node: ast.expr) -> bool:
    """`<anything>.build()` or `<builder>.simple(...)`: the two ways the builder returns a prompt."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"build", "simple"}
    )


def prompt_argument(call: ast.Call) -> ast.expr:
    for kw in call.keywords:
        if kw.arg == "prompt":
            return kw.value
    return call.args[1]  # (task_type, prompt, ...)


def is_gateway_call(call: ast.Call) -> bool:
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr in GATEWAY_METHODS):
        return False
    owner = f.value
    return (isinstance(owner, ast.Name) and owner.id == "gateway") or (
        isinstance(owner, ast.Attribute) and owner.attr == "gateway"
    )


def call_sites() -> list[tuple[str, ast.Call, ast.AST]]:
    sites = []
    for folder in ("app", "ml"):
        for path in sorted((BACKEND / folder).rglob("*.py")):
            tree = ast.parse(path.read_text())
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    for node in ast.walk(fn):
                        if isinstance(node, ast.Call) and is_gateway_call(node):
                            sites.append((f"{path.relative_to(BACKEND)}:{node.lineno}", node, fn))
    return sites


def assigned_values(fn: ast.AST, name: str) -> list[ast.expr]:
    return [
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    ]


def test_there_are_call_sites_to_check():
    places = {where.split(":")[0] for where, _, _ in call_sites()}
    assert {
        "app/api/v1/chat.py",
        "app/api/v1/experiments.py",
        "ml/agents/cleaning_agent.py",
        "ml/agents/decision_agent.py",
        "ml/experiments/planner.py",
    } <= places


def test_every_gateway_call_sends_a_built_prompt():
    bad = []
    for where, call, fn in call_sites():
        arg = prompt_argument(call)
        if isinstance(arg, ast.Name):
            values = assigned_values(fn, arg.id)
            ok = bool(values) and all(builds_prompt(v) for v in values)
        else:
            ok = builds_prompt(arg)
        if not ok:
            bad.append(where)
    assert bad == [], f"prompt not produced by the context builder at: {bad}"
