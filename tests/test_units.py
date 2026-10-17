# pyright: reportUnnecessaryTypeIgnoreComment=true
"""The unit rule table (D11): what combines, what fails, and how annotations resolve. The
``Compounding`` model below is also a pyright probe: this module is checked strictly, and the two
ignore comments on its deliberate mistakes are themselves checked for being necessary."""

from __future__ import annotations

from typing import get_args

import pytest

from halley import (
    Factor,
    Field,
    Int,
    Product,
    Schema,
    UnitError,
    annual_step,
    cc_discount,
    check,
    to_monthly,
    trace,
    var,
)
from halley.authoring import int_field
from halley.authoring.units import (
    Annual,
    Continuous,
    CoverageUnits,
    Effective,
    Index,
    Monthly,
    Prob,
    Rate,
    unit_from_annotation,
)
from halley.ir import units as u


@pytest.mark.parametrize(
    ("op", "a", "b", "expected"),
    [
        ("mul", u.COUNT, u.PROB_MONTHLY, u.COUNT),
        ("mul", u.PROB_MONTHLY, u.COUNT, u.COUNT),
        ("mul", u.COUNT, u.MONEY_PP, u.MONEY),
        ("mul", u.MONEY, u.FACTOR, u.MONEY),
        ("mul", u.MONEY_PP, u.FACTOR, u.MONEY_PP),
        ("mul", u.MONEY, u.NUMBER, u.MONEY),
        ("mul", u.RATE_MONTHLY, u.MONEY, u.MONEY),
        ("mul", u.INT, u.INTEGER, u.INT),
        ("add", u.MONEY, u.MONEY, u.MONEY),
        ("sub", u.COUNT, u.COUNT, u.COUNT),
        ("sub", u.INTEGER, u.PROB_ANNUAL, u.PROB_ANNUAL),
        ("add", u.INTEGER, u.RATE_MONTHLY, u.FACTOR),
        ("add", u.INTEGER, u.RATE_ANNUAL, u.FACTOR),
        ("sub", u.RATE_ANNUAL, u.NUMBER, u.FACTOR),
        ("add", u.INT, u.INTEGER, u.INT),
        ("truediv", u.MONEY, u.COUNT, u.MONEY_PP),
        ("truediv", u.MONEY, u.MONEY, u.FACTOR),
        ("truediv", u.MONEY, u.NUMBER, u.MONEY),
        ("truediv", u.NUMBER, u.INT, u.FACTOR),
        ("truediv", u.RATE_ANNUAL, u.INTEGER, u.RATE_ANNUAL),
        ("floordiv", u.INT, u.INTEGER, u.INT),
        ("pow", u.FACTOR, u.INT, u.FACTOR),
        ("pow", u.FACTOR, u.NUMBER, u.FACTOR),
        ("pow", u.PROB_ANNUAL, u.NUMBER, u.PROB_ANNUAL),
        ("lt", u.INT, u.INTEGER, u.BOOL),
        ("gt", u.MONEY, u.INTEGER, u.BOOL),
        ("eq", u.INT, u.INT, u.BOOL),
        ("and", u.BOOL, u.BOOL, u.BOOL),
        # Index: ratios and differences of levels, scaling, comparison with a literal
        ("truediv", u.INDEX, u.INDEX, u.FACTOR),
        ("add", u.INDEX, u.INDEX, u.INDEX),
        ("sub", u.INDEX, u.INDEX, u.INDEX),
        ("mul", u.NUMBER, u.INDEX, u.INDEX),
        ("mul", u.INDEX, u.FACTOR, u.INDEX),
        ("truediv", u.INDEX, u.INTEGER, u.INDEX),
        ("truediv", u.INDEX, u.FACTOR, u.INDEX),
        ("gt", u.INDEX, u.NUMBER, u.BOOL),
        # CoverageUnits: sums, scaling, ratios
        ("add", u.COVERAGE_UNITS, u.COVERAGE_UNITS, u.COVERAGE_UNITS),
        ("sub", u.COVERAGE_UNITS, u.COVERAGE_UNITS, u.COVERAGE_UNITS),
        ("mul", u.FACTOR, u.COVERAGE_UNITS, u.COVERAGE_UNITS),
        ("mul", u.COVERAGE_UNITS, u.NUMBER, u.COVERAGE_UNITS),
        ("truediv", u.COVERAGE_UNITS, u.INTEGER, u.COVERAGE_UNITS),
        ("truediv", u.COVERAGE_UNITS, u.FACTOR, u.COVERAGE_UNITS),
        ("truediv", u.COVERAGE_UNITS, u.COVERAGE_UNITS, u.FACTOR),
        # a continuously compounded rate scales and divides like any rate
        ("truediv", u.RATE_ANNUAL_CC, u.FACTOR, u.RATE_ANNUAL_CC),
        ("truediv", u.RATE_ANNUAL_CC, u.INTEGER, u.RATE_ANNUAL_CC),
        ("mul", u.NUMBER, u.RATE_ANNUAL_CC, u.RATE_ANNUAL_CC),
        ("add", u.RATE_ANNUAL_CC, u.RATE_ANNUAL_CC, u.RATE_ANNUAL_CC),
    ],
)
def test_legal_combinations(op: str, a: u.Unit, b: u.Unit, expected: u.Unit) -> None:
    assert u.binary_unit(op, a, b) == expected


@pytest.mark.parametrize(
    ("op", "a", "b", "hint"),
    [
        ("mul", u.COUNT, u.PROB_ANNUAL, "to_monthly_prob"),
        ("mul", u.COUNT, u.PROB, "untagged Prob"),
        ("mul", u.MONEY, u.COUNT, "twice"),
        ("mul", u.MONEY, u.MONEY, "no meaning"),
        ("mul", u.COUNT, u.COUNT, "no meaning"),
        ("mul", u.MONEY, u.RATE_ANNUAL, "to_monthly"),
        ("add", u.MONEY, u.MONEY_PP, "per-policy"),
        ("add", u.MONEY, u.COUNT, "mismatch"),
        ("add", u.MONEY, u.NUMBER, "mismatch"),
        ("sub", u.COUNT, u.INTEGER, "mismatch"),
        ("truediv", u.NUMBER, u.MONEY, "mismatch"),
        ("truediv", u.MONEY_PP, u.COUNT, "mismatch"),
        ("pow", u.MONEY, u.INTEGER, "mismatch"),
        ("floordiv", u.MONEY, u.INTEGER, "mismatch"),
        ("lt", u.MONEY, u.COUNT, "mismatch"),
        ("and", u.BOOL, u.INT, "mismatch"),
        # coverage units are not money
        ("add", u.MONEY, u.COVERAGE_UNITS, "coverage units are not money; convert with as_cu"),
        ("sub", u.COVERAGE_UNITS, u.MONEY, "coverage units are not money; convert with as_cu"),
        ("mul", u.COVERAGE_UNITS, u.COUNT, "mismatch"),
        ("add", u.COVERAGE_UNITS, u.FACTOR, "mismatch"),
        ("truediv", u.MONEY, u.COVERAGE_UNITS, "mismatch"),
        # a continuously compounded rate has no 1 + rate
        ("add", u.INTEGER, u.RATE_ANNUAL_CC, "accumulates through cc_discount"),
        ("add", u.RATE_ANNUAL_CC, u.NUMBER, "accumulates through cc_discount"),
        ("sub", u.NUMBER, u.RATE_ANNUAL_CC, "accumulates through cc_discount"),
        ("mul", u.RATE_ANNUAL_CC, u.FACTOR, "mismatch"),
        ("add", u.RATE_ANNUAL_CC, u.RATE_ANNUAL, "mismatch"),
        # an index is a level: nothing but ratios, differences and scaling
        ("mul", u.INDEX, u.INDEX, "mismatch"),
        ("mul", u.INDEX, u.COUNT, "mismatch"),
        ("mul", u.INDEX, u.MONEY, "mismatch"),
        ("add", u.INDEX, u.FACTOR, "mismatch"),
        ("add", u.INDEX, u.NUMBER, "mismatch"),
        ("pow", u.INDEX, u.INTEGER, "mismatch"),
        ("truediv", u.NUMBER, u.INDEX, "mismatch"),
        ("truediv", u.MONEY, u.INDEX, "mismatch"),
    ],
)
def test_illegal_combinations(op: str, a: u.Unit, b: u.Unit, hint: str) -> None:
    with pytest.raises(UnitError, match=hint):
        u.binary_unit(op, a, b)


def test_unit_sets() -> None:
    assert {u.RATE_ANNUAL, u.RATE_MONTHLY} == u.RATES_EFFECTIVE
    assert u.RATES_EFFECTIVE | {u.RATE_ANNUAL_CC} == u.RATES
    assert {u.INDEX, u.COVERAGE_UNITS, u.RATE_ANNUAL_CC} <= u.SCALED
    assert u.COVERAGE_UNITS in u.SUMMABLE
    assert u.INDEX not in u.SUMMABLE
    assert str(u.RATE_ANNUAL) == "Rate[Annual]"
    assert str(u.RATE_ANNUAL_CC) == "Rate[Annual, Continuous]"


@pytest.mark.parametrize(
    ("fn", "args", "expected"),
    [
        ("to_monthly", (u.RATE_ANNUAL,), u.RATE_MONTHLY),
        ("to_monthly", (u.NUMBER,), u.RATE_MONTHLY),
        ("to_monthly_prob", (u.PROB_ANNUAL,), u.PROB_MONTHLY),
        ("to_monthly_cf", (u.PROB_ANNUAL,), u.PROB_MONTHLY),
        ("annual_step", (u.RATE_ANNUAL,), u.FACTOR),
        ("cc_discount", (u.RATE_ANNUAL_CC, u.INT), u.FACTOR),
        ("cc_discount", (u.RATE_ANNUAL_CC, u.INTEGER), u.FACTOR),
        ("fwd_rate", (u.FACTOR, u.FACTOR), u.RATE_MONTHLY),
        ("ann_fwd", (u.FACTOR, u.FACTOR), u.RATE_ANNUAL),
        ("ann_spot", (u.FACTOR, u.INT), u.RATE_ANNUAL),
        ("as_cu", (u.MONEY,), u.COVERAGE_UNITS),
        ("abs", (u.MONEY,), u.MONEY),
        ("abs", (u.INT,), u.INT),
        ("abs", (u.RATE_ANNUAL_CC,), u.RATE_ANNUAL_CC),
        ("exp", (u.FACTOR,), u.FACTOR),
        ("minimum", (u.MONEY, u.NUMBER), u.MONEY),
    ],
)
def test_call_units(fn: str, args: tuple[u.Unit, ...], expected: u.Unit) -> None:
    assert u.call_unit(fn, args) == expected


@pytest.mark.parametrize(
    ("fn", "args", "message"),
    [
        ("to_monthly", (u.RATE_ANNUAL_CC,), r"converts an effective Rate\[Annual\], not a"),
        ("to_monthly", (u.RATE_MONTHLY,), r"converts an effective Rate\[Annual\], not a"),
        ("annual_step", (u.RATE_ANNUAL_CC,), r"converts an effective Rate\[Annual\], not a"),
        ("to_monthly_cf", (u.PROB_MONTHLY,), r"converts a Prob\[Annual\], not a"),
        ("cc_discount", (u.RATE_ANNUAL, u.INT), r"discounts with a Rate\[Annual, Continuous\]"),
        ("cc_discount", (u.RATE_ANNUAL_CC, u.NUMBER), r"the months in cc_discount\(\) is an Int"),
        ("fwd_rate", (u.MONEY, u.FACTOR), r"takes two discount factors"),
        ("ann_spot", (u.FACTOR, u.FACTOR), r"the month in ann_spot\(\) is an Int"),
        ("as_cu", (u.COUNT,), r"converts Money to coverage units, not a Count"),
        ("abs", (u.BOOL,), r"abs\(\) takes a quantity"),
        ("exp", (u.INDEX,), r"dimensionless Factor"),
        ("nonesuch", (u.FACTOR,), r"unknown function"),
    ],
)
def test_call_unit_mistakes(fn: str, args: tuple[u.Unit, ...], message: str) -> None:
    with pytest.raises(UnitError, match=message):
        u.call_unit(fn, args)


def test_where_literal_branch_takes_the_other_units_unit() -> None:
    assert u.where_unit(u.BOOL, u.MONEY, u.INTEGER) == u.MONEY
    assert u.where_unit(u.BOOL, u.NUMBER, u.PROB_MONTHLY) == u.PROB_MONTHLY
    assert u.where_unit(u.BOOL, u.INT, u.INTEGER) == u.INT
    with pytest.raises(UnitError, match="branches"):
        u.where_unit(u.BOOL, u.MONEY, u.COUNT)
    with pytest.raises(UnitError, match="Bool condition"):
        u.where_unit(u.INT, u.MONEY, u.MONEY)


def test_declared_unit_accepts_literals_only_as_adopted() -> None:
    assert u.declared_unit_ok(u.MONEY, u.MONEY)
    assert u.declared_unit_ok(u.MONEY, u.NUMBER)
    assert u.declared_unit_ok(u.INT, u.INTEGER)
    assert not u.declared_unit_ok(u.MONEY, u.COUNT)
    assert not u.declared_unit_ok(u.BOOL, u.NUMBER)


def test_annotations_resolve_to_tags() -> None:
    assert unit_from_annotation(Prob[Annual]) == u.PROB_ANNUAL
    assert unit_from_annotation(Prob[Monthly]) == u.PROB_MONTHLY
    assert unit_from_annotation(Rate[Monthly]) == u.RATE_MONTHLY
    assert unit_from_annotation(Prob) == u.PROB
    assert unit_from_annotation(Index) == u.INDEX
    assert unit_from_annotation(CoverageUnits) == u.COVERAGE_UNITS
    with pytest.raises(UnitError, match="period"):
        unit_from_annotation(Rate)
    with pytest.raises(UnitError, match="not a Halley unit"):
        unit_from_annotation(float)


def test_rate_compounding_tags() -> None:
    """``Rate[Annual]`` is the effective rate and one Unit object, so hashes and messages that
    print it are unchanged; ``Rate[Annual, Effective]`` is a second spelling of it. The PEP 696
    default is filled into ``__args__``, so ``Rate[Monthly, Effective]`` is the very annotation
    ``Rate[Monthly]`` is and resolves the same (a recorded deviation from the design's "takes no
    second tag": the resolver cannot tell the two apart); ``Rate[Monthly, Continuous]`` fails."""
    assert unit_from_annotation(Rate[Annual]) is u.RATE_ANNUAL
    assert unit_from_annotation(Rate[Annual, Effective]) is u.RATE_ANNUAL
    assert unit_from_annotation(Rate[Annual, Continuous]) is u.RATE_ANNUAL_CC
    assert unit_from_annotation(Rate[Monthly]) is u.RATE_MONTHLY
    assert get_args(Rate[Monthly]) == get_args(Rate[Monthly, Effective])
    assert unit_from_annotation(Rate[Monthly, Effective]) is u.RATE_MONTHLY
    with pytest.raises(UnitError, match="effective by definition"):
        unit_from_annotation(Rate[Monthly, Continuous])


def test_unit_names_round_trip() -> None:
    for unit in u.ALL_UNITS:
        assert u.unit_from_name(unit.name) == unit
    assert u.unit_from_name("Fraction") == u.FACTOR
    assert u.unit_from_name("PerPolicy[Money]") == u.MONEY_PP
    assert u.unit_from_name("Rate[Annual, Effective]") is u.RATE_ANNUAL
    assert u.unit_from_name("Rate[Annual, Continuous]") is u.RATE_ANNUAL_CC
    assert u.unit_from_name("Index") is u.INDEX
    assert u.unit_from_name("CoverageUnits") is u.COVERAGE_UNITS
    with pytest.raises(ValueError, match="unknown unit"):
        u.unit_from_name("Parsecs")


# ------------------------------------------------------------------------- the pyright probe

COMPOUNDING_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("esc", "number", "Rate[Annual]", ge=0),
        Field("ilp", "number", "Rate[Annual, Continuous]", ge=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


class Compounding(Product):
    """Every Rate spelling as an annotation, and the operations each allows. This module is
    type-checked strictly, so the class is a pyright probe of the overloads."""

    model_point = COMPOUNDING_SCHEMA

    @var
    def i(self) -> Rate[Annual]:
        return self.mp.esc

    @var
    def i_eff(self) -> Rate[Annual, Effective]:
        return self.i()

    @var
    def ilp(self) -> Rate[Annual, Continuous]:
        return self.mp.ilp

    @var
    def i_m(self) -> Rate[Monthly]:
        return to_monthly(self.i_eff())

    @var
    def acc(self) -> Factor:
        return (1 + self.i()) * annual_step(self.i_eff()) * (1 + self.i_m())

    @var
    def df(self, t: Int) -> Factor:
        return cc_discount(self.ilp(), months=t) * (1 + self.i_m()) ** (-t)

    @var
    def ilp_half(self) -> Rate[Annual, Continuous]:
        return self.ilp() * 0.5


class BadSum(Compounding):
    @var
    def acc(self) -> Factor:
        return 1 + self.ilp()  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


class BadConversion(Compounding):
    @var
    def i_m(self) -> Rate[Monthly]:
        return to_monthly(self.ilp())  # pyright: ignore[reportArgumentType]


def test_every_rate_spelling_builds_and_the_continuous_mistakes_fail() -> None:
    g = check(trace(Compounding))
    assert g.variables["i"].unit is u.RATE_ANNUAL
    assert g.variables["i_eff"].unit is u.RATE_ANNUAL
    assert g.variables["ilp"].unit is u.RATE_ANNUAL_CC
    assert g.variables["i_m"].unit is u.RATE_MONTHLY
    assert g.fields["ilp"].unit is u.RATE_ANNUAL_CC
    with pytest.raises(UnitError, match="accumulates through cc_discount") as info:
        check(trace(BadSum))
    assert info.value.loc is not None
    assert info.value.loc.function.endswith("BadSum.acc")
    with pytest.raises(UnitError, match=r"to_monthly\(\) converts an effective Rate\[Annual\]"):
        check(trace(BadConversion))
