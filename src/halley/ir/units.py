"""Unit tags and the unit rule table (D11).

The tags here are what the checker attaches to every node of the graph. The classes a model
writes as annotations (``Money``, ``Count``, ``Prob[Monthly]`` ...) live in
:mod:`halley.authoring.units` and map onto these tags. The rule table below is the one place that
says which combinations are legal; the operator overloads on the annotation classes repeat it for
the type checker.

v1 units (D11): ``Money``, ``MoneyPerPolicy``, ``Count``, ``Prob`` with a period tag,
``Rate[Annual]`` and ``Rate[Monthly]``. The dimensionless ``Factor`` (also written ``Fraction``)
and the exact types ``Int`` and ``Bool`` are the extensions the term assurance spec needs (its
section 15.4). The annuity spec adds ``Index`` (a price index level), ``CoverageUnits`` (the
IFRS 17 coverage units, which are not money) and the compounding tag on annual rates:
``Rate[Annual]`` is the effective rate (also written ``Rate[Annual, Effective]``), and
``Rate[Annual, Continuous]`` the continuously compounded one, which accumulates only through
``cc_discount()``. A unit mismatch is always an error, never a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from halley.errors import Loc, UnitError


@dataclass(frozen=True, slots=True)
class Unit:
    """The unit tag of a node."""

    name: str

    def __str__(self) -> str:
        return self.name


MONEY: Final = Unit("Money")
MONEY_PP: Final = Unit("MoneyPerPolicy")
COUNT: Final = Unit("Count")
PROB: Final = Unit("Prob")
PROB_ANNUAL: Final = Unit("Prob[Annual]")
PROB_MONTHLY: Final = Unit("Prob[Monthly]")
RATE_ANNUAL: Final = Unit("Rate[Annual]")
"""The effective annual rate. ``Rate[Annual, Effective]`` is a second spelling of this tag."""
RATE_ANNUAL_CC: Final = Unit("Rate[Annual, Continuous]")
"""A continuously compounded annual rate (a force): ``1 + rate`` has no meaning for it."""
RATE_MONTHLY: Final = Unit("Rate[Monthly]")
FACTOR: Final = Unit("Factor")
INDEX: Final = Unit("Index")
"""A price index level (CPI). Only ratios and differences of two levels have a meaning."""
COVERAGE_UNITS: Final = Unit("CoverageUnits")
"""IFRS 17 coverage units: a quantity of service, not money; ``as_cu()`` converts an amount."""
INT: Final = Unit("Int")
BOOL: Final = Unit("Bool")
NUMBER: Final = Unit("number")
"""A unitless float literal. It takes the unit of what it is combined with."""
INTEGER: Final = Unit("integer")
"""A unitless int literal."""
ENUM: Final = Unit("enum")
"""A string value: a table key or the operand of an equality test, nothing else."""

PROBS: Final = frozenset({PROB, PROB_ANNUAL, PROB_MONTHLY})
RATES_EFFECTIVE: Final = frozenset({RATE_ANNUAL, RATE_MONTHLY})
"""Effective rates: ``1 + rate`` is their accumulation factor."""
RATES: Final = RATES_EFFECTIVE | {RATE_ANNUAL_CC}
LITERALS: Final = frozenset({NUMBER, INTEGER})
SCALED: Final = frozenset({MONEY, MONEY_PP, COUNT, FACTOR, INDEX, COVERAGE_UNITS} | RATES | PROBS)
"""Units a dimensionless factor or a literal scales without changing them."""
INT_LIKE: Final = frozenset({INT, INTEGER})
SUMMABLE: Final = frozenset({MONEY, MONEY_PP, COUNT, FACTOR, COVERAGE_UNITS})
"""Units ``sum_t()`` adds over time. An ``Index`` level is not summed."""

ALL_UNITS: Final = (
    MONEY,
    MONEY_PP,
    COUNT,
    PROB,
    PROB_ANNUAL,
    PROB_MONTHLY,
    RATE_ANNUAL,
    RATE_ANNUAL_CC,
    RATE_MONTHLY,
    FACTOR,
    INDEX,
    COVERAGE_UNITS,
    INT,
    BOOL,
    ENUM,
)


def is_literal(u: Unit) -> bool:
    return u in LITERALS


def _lit_result(a: Unit, b: Unit) -> Unit:
    return INTEGER if a == INTEGER and b == INTEGER else NUMBER


_PRODUCTS: Final[dict[frozenset[Unit], Unit]] = {
    frozenset({COUNT, PROB_MONTHLY}): COUNT,
    frozenset({COUNT, MONEY_PP}): MONEY,
    frozenset({COUNT, FACTOR}): COUNT,
    frozenset({MONEY, FACTOR}): MONEY,
    frozenset({MONEY_PP, FACTOR}): MONEY_PP,
    frozenset({FACTOR}): FACTOR,
    frozenset({PROB, FACTOR}): PROB,
    frozenset({PROB_ANNUAL, FACTOR}): PROB_ANNUAL,
    frozenset({PROB_MONTHLY, FACTOR}): PROB_MONTHLY,
    frozenset({RATE_ANNUAL, FACTOR}): RATE_ANNUAL,
    frozenset({RATE_MONTHLY, FACTOR}): RATE_MONTHLY,
    frozenset({RATE_MONTHLY, MONEY}): MONEY,
    frozenset({RATE_MONTHLY, MONEY_PP}): MONEY_PP,
    frozenset({PROB_MONTHLY, MONEY_PP}): MONEY_PP,
    frozenset({INDEX, FACTOR}): INDEX,
    frozenset({COVERAGE_UNITS, FACTOR}): COVERAGE_UNITS,
    frozenset({INT}): INT,
}

_QUOTIENTS: Final[dict[tuple[Unit, Unit], Unit]] = {
    (MONEY, COUNT): MONEY_PP,
    (MONEY, MONEY): FACTOR,
    (MONEY_PP, MONEY_PP): FACTOR,
    (COUNT, COUNT): FACTOR,
    (FACTOR, FACTOR): FACTOR,
    (INT, INT): FACTOR,
    (INDEX, INDEX): FACTOR,
    (COVERAGE_UNITS, COVERAGE_UNITS): FACTOR,
    (MONEY, FACTOR): MONEY,
    (MONEY_PP, FACTOR): MONEY_PP,
    (COUNT, FACTOR): COUNT,
    (PROB, FACTOR): PROB,
    (PROB_ANNUAL, FACTOR): PROB_ANNUAL,
    (PROB_MONTHLY, FACTOR): PROB_MONTHLY,
    (RATE_ANNUAL, FACTOR): RATE_ANNUAL,
    (RATE_ANNUAL_CC, FACTOR): RATE_ANNUAL_CC,
    (RATE_MONTHLY, FACTOR): RATE_MONTHLY,
    (INDEX, FACTOR): INDEX,
    (COVERAGE_UNITS, FACTOR): COVERAGE_UNITS,
}

_CC_HINT: Final = "a continuously compounded rate accumulates through cc_discount()"
_CU_HINT: Final = "coverage units are not money; convert with as_cu()"

_HINTS: Final[dict[tuple[str, frozenset[Unit]], str]] = {
    ("mul", frozenset({COUNT, PROB_ANNUAL})): "an annual probability cannot multiply a count in "
    "a monthly step; convert it with to_monthly_prob()",
    ("mul", frozenset({COUNT, PROB})): "an untagged Prob cannot multiply a count; declare it "
    "Prob[Monthly] or convert it with to_monthly_prob()",
    ("mul", frozenset({MONEY, COUNT})): "Money * Count would apply the in-force count twice; "
    "multiply a per-policy amount by the count exactly once",
    ("mul", frozenset({MONEY, MONEY_PP})): "an in-force total cannot multiply a per-policy amount",
    ("mul", frozenset({MONEY})): "Money * Money has no meaning",
    ("mul", frozenset({COUNT})): "Count * Count has no meaning",
    ("mul", frozenset({COUNT, RATE_ANNUAL})): "convert the annual rate with to_monthly() first",
    ("mul", frozenset({MONEY, RATE_ANNUAL})): "convert the annual rate with to_monthly() first",
    ("mul", frozenset({MONEY_PP, RATE_ANNUAL})): "convert the annual rate with to_monthly() first",
    ("add", frozenset({MONEY, MONEY_PP})): "an in-force total and a per-policy amount cannot be "
    "added",
    ("sub", frozenset({MONEY, MONEY_PP})): "an in-force total and a per-policy amount cannot be "
    "subtracted",
    ("add", frozenset({MONEY, COVERAGE_UNITS})): _CU_HINT,
    ("sub", frozenset({MONEY, COVERAGE_UNITS})): _CU_HINT,
    ("add", frozenset({NUMBER, RATE_ANNUAL_CC})): _CC_HINT,
    ("add", frozenset({INTEGER, RATE_ANNUAL_CC})): _CC_HINT,
    ("sub", frozenset({NUMBER, RATE_ANNUAL_CC})): _CC_HINT,
    ("sub", frozenset({INTEGER, RATE_ANNUAL_CC})): _CC_HINT,
}

_SYMBOLS: Final[dict[str, str]] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "and": "&",
    "or": "|",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}


def _fail(op: str, a: Unit, b: Unit, loc: Loc | None) -> UnitError:
    hint = _HINTS.get((op, frozenset({a, b})))
    tail = f": {hint}" if hint else ""
    return UnitError(f"unit mismatch: {a} {_SYMBOLS.get(op, op)} {b}{tail}", loc)


def binary_unit(op: str, a: Unit, b: Unit, loc: Loc | None = None) -> Unit:
    """The unit of ``a <op> b``, or a :class:`~halley.errors.UnitError`."""
    if op in ("add", "sub"):
        if a == b and a not in (BOOL, ENUM):
            return a
        if is_literal(a) and is_literal(b):
            return _lit_result(a, b)
        if {a, b} == {INT, INTEGER}:
            return INT
        if is_literal(a) or is_literal(b):
            other = b if is_literal(a) else a
            if other == FACTOR or other in PROBS:
                return other
            if other in RATES_EFFECTIVE:
                return FACTOR  # 1 + i and 1 + i_m are accumulation factors (term spec 15.4 (c))
        raise _fail(op, a, b, loc)
    if op == "mul":
        if is_literal(a) and is_literal(b):
            return _lit_result(a, b)
        if is_literal(a) or is_literal(b):
            lit, other = (a, b) if is_literal(a) else (b, a)
            if other in SCALED:
                return other
            if other == INT:
                return INT if lit == INTEGER else FACTOR
            raise _fail(op, a, b, loc)
        result = _PRODUCTS.get(frozenset({a, b}))
        if result is None:
            raise _fail(op, a, b, loc)
        return result
    if op == "truediv":
        if is_literal(a) and is_literal(b):
            return NUMBER
        if is_literal(b) or b == INT:
            if is_literal(a):
                return FACTOR
            if a in SCALED:
                return a  # dividing by a plain number or a month count keeps the unit
            if a == INT:
                return FACTOR
            raise _fail(op, a, b, loc)
        if is_literal(a):
            if b in (FACTOR, INT):
                return FACTOR
            raise _fail(op, a, b, loc)
        result = _QUOTIENTS.get((a, b))
        if result is None:
            raise _fail(op, a, b, loc)
        return result
    if op in ("floordiv", "mod"):
        if a in INT_LIKE and b in INT_LIKE:
            return INTEGER if a == INTEGER and b == INTEGER else INT
        raise _fail(op, a, b, loc)
    if op == "pow":
        if is_literal(a) and is_literal(b):
            return NUMBER
        if a == FACTOR and (is_literal(b) or b == INT):
            return FACTOR
        if a in PROBS and is_literal(b):
            return a
        if is_literal(a) and b == INT:
            return FACTOR
        raise _fail(op, a, b, loc)
    if op in ("lt", "le", "gt", "ge", "eq", "ne"):
        if a == b and a != BOOL:
            return BOOL
        if a in (BOOL, ENUM) or b in (BOOL, ENUM):
            raise _fail(op, a, b, loc)
        if is_literal(a) or is_literal(b):
            return BOOL
        raise _fail(op, a, b, loc)
    if op in ("and", "or"):
        if a == BOOL and b == BOOL:
            return BOOL
        raise _fail(op, a, b, loc)
    raise UnitError(f"unknown operator {op!r}", loc)


def unary_unit(op: str, a: Unit, loc: Loc | None = None) -> Unit:
    if op == "neg":
        if a in (BOOL, ENUM):
            raise UnitError(f"unit mismatch: -{a}", loc)
        return a
    if op == "not":
        if a != BOOL:
            raise UnitError(f"unit mismatch: ~{a}", loc)
        return BOOL
    raise UnitError(f"unknown operator {op!r}", loc)


def branch_unit(a: Unit, b: Unit, loc: Loc | None, what: str) -> Unit:
    """The unit of two alternatives: a literal takes the other branch's unit, except that a
    float literal is not an Int (an Int holds whole numbers: a month, a key, a count of
    months)."""
    if a == b:
        return a
    if is_literal(a) and is_literal(b):
        return _lit_result(a, b)
    if {a, b} == {INT, INTEGER}:
        return INT
    if {a, b} == {INT, NUMBER}:
        raise UnitError(
            f"unit mismatch: the branches of {what} are {a} and {b}: an Int holds whole "
            "numbers, so write the literal as an int",
            loc,
        )
    if is_literal(a) and b != BOOL:
        return b
    if is_literal(b) and a != BOOL:
        return a
    raise UnitError(f"unit mismatch: the branches of {what} are {a} and {b}", loc)


def where_unit(cond: Unit, a: Unit, b: Unit, loc: Loc | None = None) -> Unit:
    if cond != BOOL:
        raise UnitError(f"where() needs a Bool condition, not {cond}", loc)
    return branch_unit(a, b, loc, "where()")


def _takes(fn: str, arg: Unit, wanted: Unit, what: str, loc: Loc | None) -> None:
    """``fn`` converts a ``wanted`` quantity (or a literal, which adopts it)."""
    if arg != wanted and not is_literal(arg):
        raise UnitError(f"{fn}() {what}, not a {arg}", loc)


def _month_arg(fn: str, arg: Unit, what: str, loc: Loc | None) -> None:
    if arg not in INT_LIKE:
        raise UnitError(f"the {what} in {fn}() is an Int, not a {arg}", loc)


def call_unit(fn: str, args: tuple[Unit, ...], loc: Loc | None = None) -> Unit:
    """The unit of a formula function applied to arguments of the given units."""
    if fn in ("exp", "log", "expm1", "log1p", "sqrt"):
        (a,) = args
        if a == FACTOR or is_literal(a):
            return FACTOR
        raise UnitError(f"{fn}() takes a dimensionless Factor, not {a}", loc)
    if fn in ("minimum", "maximum"):
        a, b = args
        return branch_unit(a, b, loc, f"{fn}()")
    if fn == "abs":
        (a,) = args
        if a in SCALED or a == INT or is_literal(a):
            return a
        raise UnitError(f"abs() takes a quantity, not a {a}", loc)
    if fn == "to_monthly":
        (a,) = args
        _takes(fn, a, RATE_ANNUAL, "converts an effective Rate[Annual]", loc)
        return RATE_MONTHLY
    if fn == "to_monthly_prob":
        (a,) = args
        _takes(fn, a, PROB_ANNUAL, "converts a Prob[Annual]", loc)
        return PROB_MONTHLY
    if fn == "to_monthly_cf":
        (a,) = args
        _takes(fn, a, PROB_ANNUAL, "converts a Prob[Annual]", loc)
        return PROB_MONTHLY
    if fn == "annual_step":
        (a,) = args
        _takes(fn, a, RATE_ANNUAL, "converts an effective Rate[Annual]", loc)
        return FACTOR
    if fn == "cc_discount":
        s, m = args
        _takes(fn, s, RATE_ANNUAL_CC, "discounts with a Rate[Annual, Continuous]", loc)
        _month_arg(fn, m, "months", loc)
        return FACTOR
    if fn in ("fwd_rate", "ann_fwd"):
        p0, p1 = args
        for p in (p0, p1):
            _takes(fn, p, FACTOR, "takes two discount factors (Factor)", loc)
        return RATE_MONTHLY if fn == "fwd_rate" else RATE_ANNUAL
    if fn == "ann_spot":
        p, m = args
        _takes(fn, p, FACTOR, "takes a discount factor (Factor)", loc)
        _month_arg(fn, m, "month", loc)
        return RATE_ANNUAL
    if fn == "as_cu":
        (a,) = args
        _takes(fn, a, MONEY, "converts Money to coverage units", loc)
        return COVERAGE_UNITS
    raise UnitError(f"unknown function {fn}()", loc)


def declared_unit_ok(declared: Unit, inferred: Unit) -> bool:
    """Whether a formula's inferred unit satisfies the declared return unit: a literal takes
    the declared unit, except that a float literal is not an Int (:func:`branch_unit`)."""
    if declared == inferred:
        return True
    if is_literal(inferred):
        if declared == INT:
            return inferred == INTEGER
        return declared not in (BOOL, ENUM)
    return False


def unit_from_name(name: str) -> Unit:
    """The tag for a unit written as a string in a schema or table definition."""
    table: dict[str, Unit] = {u.name: u for u in ALL_UNITS}
    table["Fraction"] = FACTOR
    table["PerPolicy[Money]"] = MONEY_PP
    table["Rate[Annual, Effective]"] = RATE_ANNUAL
    if name not in table:
        raise ValueError(f"unknown unit {name!r}; one of {', '.join(sorted(table))}")
    return table[name]
