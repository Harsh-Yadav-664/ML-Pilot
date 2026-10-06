"""The one safe evaluator for engineered feature formulas.

LLM-proposed formulas are data, never code: a formula is parsed with `ast`,
checked against a whitelist and fixed limits, and then evaluated by walking the
tree with pandas/NumPy operations. Nothing is ever passed to eval/exec.

    expr = parse("log1p(income) / (age + 1)", df.columns)   # validation only, no data
    values = evaluate(expr, df)                               # a float Series

Allowed: numeric constants, column names (a string constant also names a column,
for names that are not identifiers), + - * / // % **, comparisons, & | ~ (and
and/or/not, elementwise), and calls to the functions in FUNCTIONS. Anything else
raises InvalidFormula. Division by zero gives NaN; infinities become NaN.

This file is also copied verbatim into exported scoring scripts, so it must only
import numpy and pandas.
"""

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAX_FORMULA_CHARS = 1000
MAX_EXPONENT = 8
MAX_CONSTANT = 1e12
MAX_DEPTH = 20
MAX_NODES = 200


def _where(cond: Any, a: Any, b: Any) -> Any:
    return np.where(cond, a, b)


# name -> (function, number of arguments)
FUNCTIONS: dict[str, tuple[Any, int]] = {
    "log1p": (np.log1p, 1),
    "sqrt": (np.sqrt, 1),
    "abs": (np.abs, 1),
    "isnull": (pd.isna, 1),
    "minimum": (np.minimum, 2),
    "maximum": (np.maximum, 2),
    "clip": (np.clip, 3),
    "where": (_where, 3),
}

_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_LOGICAL = (ast.BitAnd, ast.BitOr)
_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class InvalidFormula(ValueError):
    """The formula is not allowed, or cannot be computed on this data."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Expr:
    """A formula that passed validation against a set of column names."""

    formula: str
    tree: ast.Expression
    columns: frozenset


def _column_of(node: ast.AST, columns: frozenset) -> str | None:
    if isinstance(node, ast.Name) and node.id in columns:
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in columns:
        return node.value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _constant_exponent(node: ast.AST) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _constant_exponent(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and _is_number(node.value):
        return float(node.value)
    return None


def _check(node: ast.AST, columns: frozenset, depth: int, used: set) -> None:
    if depth > MAX_DEPTH:
        raise InvalidFormula(f"formula is nested more than {MAX_DEPTH} levels deep")
    nxt = depth + 1

    column = _column_of(node, columns)
    if column is not None:
        used.add(column)
        return
    if isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            raise InvalidFormula(f"'{node.id}' is a function; call it, e.g. {node.id}(x)")
        raise InvalidFormula(f"unknown column '{node.id}'")
    if isinstance(node, ast.Constant):
        if not _is_number(node.value):
            raise InvalidFormula(f"constant {node.value!r} is not a number or a column name")
        if abs(node.value) > MAX_CONSTANT:
            raise InvalidFormula(f"constant {node.value} is larger than {MAX_CONSTANT:g}")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ARITHMETIC + _LOGICAL):
            raise InvalidFormula(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow):
            exponent = _constant_exponent(node.right)
            if exponent is None:
                raise InvalidFormula("the exponent of ** must be a constant number")
            if abs(exponent) > MAX_EXPONENT:
                raise InvalidFormula(f"exponent {exponent:g} is larger than {MAX_EXPONENT}")
        _check(node.left, columns, nxt, used)
        _check(node.right, columns, nxt, used)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert, ast.Not)):
            raise InvalidFormula(f"operator {type(node.op).__name__} is not allowed")
        _check(node.operand, columns, nxt, used)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value, columns, nxt, used)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, _COMPARE):
                raise InvalidFormula(f"comparison {type(op).__name__} is not allowed")
        for operand in (node.left, *node.comparators):
            _check(operand, columns, nxt, used)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            raise InvalidFormula(
                f"function '{name}' is not allowed; allowed: {', '.join(FUNCTIONS)}"
            )
        if node.keywords:
            raise InvalidFormula("keyword arguments are not allowed")
        _, arity = FUNCTIONS[node.func.id]
        if len(node.args) != arity or any(isinstance(a, ast.Starred) for a in node.args):
            raise InvalidFormula(f"{node.func.id} takes {arity} argument(s)")
        for arg in node.args:
            _check(arg, columns, nxt, used)
        return
    raise InvalidFormula(f"{type(node).__name__} is not allowed in a formula")


def parse(formula: str, columns: Iterable[str]) -> Expr:
    """Validate a formula against column names (no data is touched). Raises InvalidFormula."""
    if not isinstance(formula, str) or not formula.strip():
        raise InvalidFormula("formula is empty")
    if len(formula) > MAX_FORMULA_CHARS:
        raise InvalidFormula(f"formula is longer than {MAX_FORMULA_CHARS} characters")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise InvalidFormula(f"not a valid expression: {e}") from None
    n_nodes = sum(
        1
        for n in ast.walk(tree)
        if not isinstance(n, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop))
    )
    if n_nodes > MAX_NODES:
        raise InvalidFormula(f"formula has more than {MAX_NODES} parts")
    cols = frozenset(str(c) for c in columns)
    used: set = set()
    _check(tree.body, cols, 1, used)
    if not used:
        raise InvalidFormula("formula does not use any column")
    return Expr(formula=formula, tree=tree, columns=frozenset(used))


def _safe_divide(op: ast.operator, left: Any, right: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(op, ast.Div):
            out = np.true_divide(left, right)
        elif isinstance(op, ast.FloorDiv):
            out = np.floor_divide(left, right)
        else:
            out = np.mod(left, right)
    return np.where(np.asarray(right) == 0, np.nan, out)


def _eval(node: ast.AST, df: pd.DataFrame, columns: frozenset) -> Any:
    column = _column_of(node, columns)
    if column is not None:
        return df[column].to_numpy()
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, df, columns), _eval(node.right, df, columns)
        op = node.op
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            return _safe_divide(op, left, right)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if isinstance(op, ast.Add):
                return np.add(left, right)
            if isinstance(op, ast.Sub):
                return np.subtract(left, right)
            if isinstance(op, ast.Mult):
                return np.multiply(left, right)
            if isinstance(op, ast.Pow):
                return np.power(np.asarray(left, dtype="float64"), right)
        if isinstance(op, ast.BitAnd):
            return np.logical_and(left, right)
        return np.logical_or(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, df, columns)
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        return np.logical_not(operand)
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, df, columns) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = values[0]
        for v in values[1:]:
            out = combine(out, v)
        return out
    if isinstance(node, ast.Compare):
        compare = {
            ast.Eq: np.equal,
            ast.NotEq: np.not_equal,
            ast.Lt: np.less,
            ast.LtE: np.less_equal,
            ast.Gt: np.greater,
            ast.GtE: np.greater_equal,
        }
        left = _eval(node.left, df, columns)
        out = None
        for op, right_node in zip(node.ops, node.comparators):
            right = _eval(right_node, df, columns)
            step = compare[type(op)](left, right)
            out = step if out is None else np.logical_and(out, step)
            left = right
        return out
    if isinstance(node, ast.Call):
        func, _ = FUNCTIONS[node.func.id]
        with np.errstate(divide="ignore", invalid="ignore"):
            return func(*(_eval(a, df, columns) for a in node.args))
    raise InvalidFormula(
        f"{type(node).__name__} is not allowed in a formula"
    )  # parse() prevents this


def evaluate(expr: Expr, df: pd.DataFrame) -> pd.Series:
    """Compute a parsed formula on a frame. Returns float64; inf and division by zero become NaN."""
    missing = sorted(expr.columns - set(map(str, df.columns)))
    if missing:
        raise InvalidFormula(f"unknown column(s): {', '.join(missing)}")
    try:
        values = _eval(expr.tree.body, df, frozenset(map(str, df.columns)))
        values = np.broadcast_to(np.asarray(values, dtype="float64"), (len(df),))
    except InvalidFormula:
        raise
    except (TypeError, ValueError) as e:
        raise InvalidFormula(
            f"cannot compute on this data (are all used columns numeric?): {e}"
        ) from None
    out = pd.Series(values, index=df.index, dtype="float64")
    return out.replace([np.inf, -np.inf], np.nan)


def evaluate_formula(formula: str, df: pd.DataFrame) -> pd.Series:
    """parse() then evaluate() against the frame's own columns."""
    return evaluate(parse(formula, df.columns), df)
