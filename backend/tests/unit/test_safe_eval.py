"""The one safe evaluator: hostile formulas are rejected fast, useful ones work."""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import pytest

from ml.features.safe_eval import InvalidFormula, evaluate, evaluate_formula, parse

DF = pd.DataFrame(
    {
        "income": [0.0, 10.0, 100.0, np.nan],
        "age": [10, 20, 0, 40],
        "Monthly Charges": [1.0, 2.0, 3.0, 4.0],
        "plan": ["a", "b", "c", "a"],
    }
)

HOSTILE = [
    "__import__('os')",
    "().__class__",
    "open('x')",
    "a ** 10**9",
    "income ** 10**9",
    "income ** 9",
    "income ** age",
    "income.__class__.__mro__",
    "__builtins__",
    "eval('1')",
    "exec('import os')",
    "getattr(income, 'x')",
    "lambda: 1",
    "(lambda x: x)(income)",
    "[x for x in income]",
    "{x for x in income}",
    "(x for x in income)",
    "income[0]",
    "income if age else 1",
    "abs(x=income)",
    "abs(*income)",
    "np.log(income)",
    "f'{income}'",
    "income := 1",
    "'not a column'",
    "1e13 * income",
    "5",
    "",
    "income +",
    "income; import os",
    "-(" * 30 + "income" + ")" * 30,
    "income + " * 300 + "income",
]


@pytest.mark.parametrize("formula", HOSTILE)
def test_hostile_formulas_are_rejected_fast(formula):
    start = time.perf_counter()
    with pytest.raises(InvalidFormula):
        evaluate_formula(formula, DF.assign(a=1.0))
    assert time.perf_counter() - start < 0.05


def test_useful_formulas_work():
    out = evaluate_formula("log1p(income) / (age + 1)", DF)
    assert out.dtype == "float64" and out.index.equals(DF.index)
    np.testing.assert_allclose(out[:3], np.log1p([0.0, 10.0, 100.0]) / np.array([11, 21, 1]))
    assert np.isnan(out[3])  # missing input stays missing
    assert evaluate_formula("where(age >= 18, 1, 0)", DF).tolist() == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("income / age", [0.0, 0.5, np.nan, np.nan]),  # division by zero is NaN, not an error
        ("age // 3 + age % 3", [4.0, 8.0, 0.0, 14.0]),
        ("age % (age - 10)", [np.nan, 0.0, 0.0, 10.0]),  # modulo by zero is NaN too
        ("(age > 5) & (age < 30)", [1.0, 1.0, 0.0, 0.0]),
        ("(age < 5) | ~(age < 30)", [0.0, 0.0, 1.0, 1.0]),
        ("age > 5 and not income > 50", [1.0, 1.0, 0.0, 1.0]),
        ("5 < age <= 20", [1.0, 1.0, 0.0, 0.0]),
        ("'Monthly Charges' * 2", [2.0, 4.0, 6.0, 8.0]),  # string constant names a column
        ("clip(age, 5, 30) ** 2", [100.0, 400.0, 25.0, 900.0]),
        ("minimum(age, 15) + maximum(age, 15)", [25.0, 35.0, 15.0, 55.0]),
        ("sqrt(abs(-age))", [np.sqrt(10), np.sqrt(20), 0.0, np.sqrt(40)]),
        ("isnull(income)", [0.0, 0.0, 0.0, 1.0]),
        ("income ** -1", [np.nan, 0.1, 0.01, np.nan]),  # 1/0 = inf becomes NaN
    ],
)
def test_operators_and_functions(formula, expected):
    np.testing.assert_allclose(evaluate_formula(formula, DF), expected)


def test_parse_needs_no_data_and_reports_the_reason():
    expr = parse("log1p(income) / (age + 1)", ["income", "age"])
    assert expr.columns == {"income", "age"}
    with pytest.raises(InvalidFormula, match="unknown column 'salary'"):
        parse("salary * 2", ["income"])
    with pytest.raises(InvalidFormula, match="not allowed"):
        parse("os.system('x')", ["os"])


def test_text_column_in_arithmetic_is_rejected_not_zeroed():
    with pytest.raises(InvalidFormula, match="cannot compute"):
        evaluate_formula("plan * 2", DF)


def test_evaluate_checks_columns_of_the_frame_it_gets():
    expr = parse("income + 1", ["income"])
    with pytest.raises(InvalidFormula, match="unknown column"):
        evaluate(expr, DF.drop(columns=["income"]))


_ATOMS = [
    "income",
    "age",
    "plan",
    "x",
    "__class__",
    "1",
    "2.5",
    "'age'",
    "'x'",
    "()",
    "[]",
    "None",
    "True",
]
_WRAPS = [
    "({}) + ({})",
    "({}) * ({})",
    "({}) / ({})",
    "({}) ** ({})",
    "({}) > ({})",
    "({}) & ({})",
    "({}).__class__",
    "({})[{}]",
    "log1p({})",
    "where({}, {}, 0)",
    "open({})",
    "__import__({})",
    "getattr({}, {})",
    "lambda: {}",
    "[{} for _ in {}]",
    "-({})",
    "~({})",
    "{} if {} else 0",
    "clip({})",
    "abs({}, {})",
    "({}).mean()",
    "sqrt({})",
]


def _random_formula(rng: random.Random, depth: int) -> str:
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(_ATOMS)
    template = rng.choice(_WRAPS)
    return template.format(*(_random_formula(rng, depth - 1) for _ in range(template.count("{}"))))


def test_fuzz_only_safe_outcomes():
    """Random mixes of allowed and hostile syntax: always InvalidFormula or a float Series."""
    rng = random.Random(0)
    outcomes = {"rejected": 0, "computed": 0}
    for _ in range(3000):
        formula = _random_formula(rng, rng.randint(1, 5))
        try:
            out = evaluate_formula(formula, DF)
        except InvalidFormula:
            outcomes["rejected"] += 1
            continue
        assert isinstance(out, pd.Series) and out.dtype == "float64" and len(out) == len(DF), (
            formula
        )
        assert not np.isinf(out).any(), formula
        outcomes["computed"] += 1
    assert outcomes["rejected"] > 0 and outcomes["computed"] > 0
