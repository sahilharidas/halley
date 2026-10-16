"""The unit rule table (D11): what combines, what fails, and how annotations resolve."""

from __future__ import annotations

import pytest

from halley.authoring.units import Annual, Monthly, Prob, Rate, unit_from_annotation
from halley.errors import UnitError
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
    ],
)
def test_illegal_combinations(op: str, a: u.Unit, b: u.Unit, hint: str) -> None:
    with pytest.raises(UnitError, match=hint):
        u.binary_unit(op, a, b)


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
    with pytest.raises(UnitError, match="period"):
        unit_from_annotation(Rate)
    with pytest.raises(UnitError, match="not a Halley unit"):
        unit_from_annotation(float)


def test_unit_names_round_trip() -> None:
    for unit in u.ALL_UNITS:
        assert u.unit_from_name(unit.name) == unit
    assert u.unit_from_name("Fraction") == u.FACTOR
    assert u.unit_from_name("PerPolicy[Money]") == u.MONEY_PP
    with pytest.raises(ValueError, match="unknown unit"):
        u.unit_from_name("Parsecs")
