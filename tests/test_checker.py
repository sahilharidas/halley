"""Checker rules, each with the deliberate mistake the term assurance spec says must fail
(section 15.3) and a few more. Every error must point at the model source."""

from __future__ import annotations

import re

import pytest

from halley import (
    Annual,
    BuildError,
    BuildWarning,
    Column,
    Count,
    Factor,
    Field,
    Int,
    Key,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    Schema,
    Table,
    UnitError,
    at,
    build,
    check,
    ir,
    sum_t,
    to_monthly_prob,
    trace,
    var,
    where,
)
from halley.authoring import int_field, maximum, minimum
from halley.ir import units as u
from halley.products.term_assurance import MORT_TABLE, RATES, SCHEMA, TermAssurance
from models import SIMPLE_SCHEMA, Expensed, Margin


def _fails(model: type[Product], match: str, error: type[Exception] = BuildError) -> Exception:
    with pytest.raises(error, match=match) as info:
        check(trace(model))
    exc = info.value
    loc = getattr(exc, "loc", None)
    assert loc is not None, "every checker error carries a model source location"
    assert loc.file.endswith(("test_checker.py", "term_assurance.py"))
    return exc


def _init(graph: ir.Graph, name: str) -> ir.Node:
    node = graph.variables[name].init
    assert node is not None
    return node


def test_term_assurance_builds_clean() -> None:
    compiled = build(TermAssurance)
    directions = {name: (v.direction, v.start) for name, v in compiled.graph.variables.items()}
    assert directions["pols_if"] == ("forward", 0)
    assert directions["deaths"] == ("forward", 1)
    assert directions["reserve"] == ("backward", 0)
    assert directions["reserve_pp"] == ("none", 0)  # at(reserve_pp, 0) makes it a state
    assert directions["disc"] == ("none", 0)  # declared a state, and read at t-1
    assert directions["dur_y"] == ("none", 1)  # a month quantity
    assert directions["v"] == ("none", 1)


def test_count_times_count_times_amount_is_a_unit_error() -> None:
    class Bad(TermAssurance):
        @var(timing="end")
        def death_outgo(self, t: Int) -> Money:
            return self.pols_if(t - 1) * self.deaths(t) * self.mp.sum_assured  # type: ignore[operator]

    _fails(Bad, "Count \\* Count", UnitError)


def test_per_policy_plus_in_force_is_a_unit_error() -> None:
    class Bad(TermAssurance):
        @var(timing="start")
        def premium_income(self, t: Int) -> Money:
            return self.pols_if(t - 1) * self.premium_pp(t) + self.premium_pp(t)  # type: ignore[operator]

    _fails(Bad, "per-policy", UnitError)


def test_declared_unit_must_match_the_formula() -> None:
    class Bad(TermAssurance):
        @var
        def net_cf(self, t: Int) -> Count:  # pyright: ignore[reportIncompatibleMethodOverride]
            return self.premium_income(t) - self.commission(t)  # type: ignore[return-value]

    _fails(Bad, "declared Count but its formula is Money", UnitError)


def test_annual_probability_divided_by_twelve_is_caught() -> None:
    class Bad(TermAssurance):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t - 1) * (self.q_ann(t) / 12)  # type: ignore[operator]

    _fails(Bad, "to_monthly_prob", UnitError)


def test_annual_rate_divided_by_twelve_is_caught() -> None:
    class Bad(TermAssurance):
        @var
        def i_m(self) -> Rate[Monthly]:
            return self.i() / 12  # type: ignore[return-value]

    _fails(Bad, "declared Rate\\[Monthly\\] but its formula is Rate\\[Annual\\]", UnitError)


def test_compounding_with_annual_rate_fails_without_conversion() -> None:
    class Bad(TermAssurance):
        @var(timing="end")
        def net_outgo(self, t: Int) -> Money:
            return self.death_outgo(t) - self.premium_income(t) * self.i()  # type: ignore[operator]

    _fails(Bad, "to_monthly", UnitError)


def test_recursive_variable_without_init_fails() -> None:
    class Bad(TermAssurance):
        @var(role="in_force")
        def pols_if(self, t: Int) -> Count:
            return self.pols_if(t - 1) - self.deaths(t) - self.lapses(t) - self.maturities(t)

    _fails(Bad, "needs an init value")


def test_backward_variable_without_terminal_fails() -> None:
    class Bad(TermAssurance):
        @var(role="reserve")
        def reserve(self, t: Int) -> Money:
            return (self.reserve(t + 1) + self.net_outgo(t + 1)) * self.v(t + 1)

    _fails(Bad, "terminal")


def test_same_period_cycle_fails() -> None:
    class Bad(TermAssurance):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t) * self.q_m(t)

    _fails(Bad, "same-period cycle")


def test_mixed_forward_backward_component_fails_d13() -> None:
    class Bad(TermAssurance):
        @var(timing="end")
        def net_outgo(self, t: Int) -> Money:
            interest = self.reserve(t - 1) * self.i_m()  # interest on the opening reserve
            return self.death_outgo(t) + self.maint_expense(t) - interest

    _fails(Bad, "mixes forward and backward")


def test_reading_the_reserve_in_the_same_month_joins_the_backward_component() -> None:
    class Design(TermAssurance):
        @var
        def surrender(self, t: Int) -> Money:
            return self.reserve(t) * 0.9

        outputs = (*TermAssurance.outputs, "surrender")  # type: ignore[misc]

    g = check(trace(Design))
    assert g.variables["surrender"].direction == "none"
    assert g.variables["surrender"].component > g.variables["reserve"].component


def test_reading_two_months_back_is_outside_the_domain() -> None:
    class Bad(TermAssurance):
        @var(role="decrement")
        def deaths(self, t: Int) -> Count:
            return self.pols_if(t - 2) * self.q_m(t)

    _fails(Bad, "before t = 0")


def test_a_month_quantity_cannot_be_demanded_at_t_zero() -> None:
    class Bad(TermAssurance):
        @var(domain="month")
        def dur_y(self, t: Int) -> Int:
            k = (self.mp.dur_m + t - 1) // 12
            return where(k < self.mp.dur_y_last, k, self.mp.dur_y_last)

        @var
        def age(self, t: Int) -> Int:
            return self.mp.age_at_entry + self.dur_y(t - 1)

    _fails(Bad, "domain='month'")


def test_a_demanded_state_must_be_computable_at_t_zero() -> None:
    class Bad(TermAssurance):
        @var
        def lagged(self, t: Int) -> Money:
            return self.death_outgo(t - 1) * 1.0

        outputs = (*TermAssurance.outputs, "lagged")  # type: ignore[misc]

    # death_outgo would be needed at t = 0, where it reads deaths(0) and pols_if(-1)
    _fails(Bad, "before t = 0")


def test_scalar_variable_cannot_read_a_time_variable_directly() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def cf(self, t: Int) -> Money:
            return self.mp.premium * self.mp.pols_if_init

        @var
        def total(self) -> Money:
            return self.cf(1)  # type: ignore[arg-type]

    with pytest.raises(BuildError, match=r"at\(cf, 1\)"):
        check(trace(Bad))

    class Bad2(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def cf(self, t: Int) -> Money:
            return self.mp.premium * self.mp.pols_if_init

        @var
        def total(self) -> Money:
            return sum_t(self.cf) + sum_t(self.cf) * 0.0

    check(trace(Bad2))


def test_table_key_units_are_checked() -> None:
    class Bad(TermAssurance):
        @var
        def q_ann(self, t: Int) -> Prob[Annual]:
            return self.tables.mort_table[self.mp.sex, self.q_m(t)]

    _fails(Bad, "Int in \\[18, 120\\]", UnitError)

    class BadConst(TermAssurance):
        @var
        def i(self) -> Rate[Annual]:
            return self.tables.rates["interest"]

    _fails(BadConst, "not a value of key name")


ENUM_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        Field("sex", "enum", enum=("M", "F")),
        Field("sex2", "enum", enum=("M", "F")),
        Field("band", "enum", enum=("A", "B", "C")),
        Field("sex_rev", "enum", enum=("F", "M")),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)
BY_SEX = Table("by_sex", keys=(Key("sex", enum=("M", "F")),), columns=(Column("v", "Factor"),))
BY_K = Table("by_k", keys=(Key("k", lo=0, hi=10),), columns=(Column("v", "Factor"),))


def test_an_enum_key_is_a_value_or_a_field_with_the_keys_values_in_order() -> None:
    """The generated code reads an enum key's table at the field's code, its position in the
    field's own enum: a field listing the key's values in another order would read the other
    row, and a where() or minimum() of fields with other values would read outside the table,
    on both backends."""

    class Reordered(Product):
        model_point = ENUM_SCHEMA
        assumption_tables = (BY_SEX,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_sex[self.mp.sex_rev] * 1.0

    _fails(
        Reordered,
        r"^mp\.sex_rev lists the values of key sex of table by_sex in another order \(F, M "
        r"against M, F\); list them in the key's order",
    )

    class OtherValues(Product):
        model_point = ENUM_SCHEMA
        assumption_tables = (BY_SEX,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_sex[where(t > 0, self.mp.band, self.mp.sex)] * 1.0

    _fails(OtherValues, r"^mp\.band does not have the same enum values as key sex of table by_sex")

    class Smallest(Product):
        model_point = ENUM_SCHEMA
        assumption_tables = (BY_SEX,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_sex[minimum(self.mp.sex, self.mp.sex2)] * 1.0

    _fails(
        Smallest,
        r"^key sex of table by_sex is an enum \(M, F\); pass one of its values, a model point "
        r"field with the same values in the same order, or a where\(\) between such fields",
    )

    class Chosen(Product):
        model_point = ENUM_SCHEMA
        assumption_tables = (BY_SEX,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_sex[where(t > 1, self.mp.sex2, self.mp.sex)] * 1.0

    check(trace(Chosen))


def test_a_float_literal_is_not_an_int() -> None:
    """An Int holds whole numbers: a float literal beside an Int in where(), minimum() or
    maximum(), or as an Int variable's value, is a unit mistake (D11), not a value truncated
    when it is stored or used as a key."""

    class Key25(Product):
        model_point = SIMPLE_SCHEMA
        assumption_tables = (BY_K,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_k[where(t > 3, self.mp.k, 2.5)]

    _fails(
        Key25,
        r"^unit mismatch: the branches of where\(\) are Int and number: an Int holds whole "
        r"numbers, so write the literal as an int",
        UnitError,
    )

    class Least(Product):
        model_point = SIMPLE_SCHEMA
        assumption_tables = (BY_K,)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_k[maximum(2.5, t)]

    _fails(Least, r"^unit mismatch: the branches of maximum\(\) are number and Int", UnitError)

    class Half(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def n(self, t: Int) -> Int:
            return 2.5  # type: ignore[return-value]

    _fails(Half, r"^n is declared Int but its formula is number", UnitError)

    class HalfInit(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=0.5)
        def n(self, t: Int) -> Int:
            return self.n(t - 1) + 1

    _fails(HalfInit, r"^the init value of n is number, not Int", UnitError)

    class Whole(Product):
        model_point = SIMPLE_SCHEMA
        assumption_tables = (BY_K,)

        @var(init=0)
        def n(self, t: Int) -> Int:
            return self.n(t - 1) + where(t > 3, self.mp.k, 2)

        @var
        def x(self, t: Int) -> Factor:
            return self.tables.by_k[minimum(self.n(t), 10)]

    check(trace(Whole))


def test_roles_are_checked() -> None:
    class TwoInForce(TermAssurance):
        @var(init=lambda mp: mp.pols_if_init, role="in_force")
        def other(self, t: Int) -> Count:
            return self.other(t - 1) * 1.0

    with pytest.raises(BuildError, match="only one variable carries role='in_force'"):
        check(trace(TwoInForce))

    class WrongUnit(TermAssurance):
        @var(role="decrement")
        def commission(self, t: Int) -> Money:
            return self.premium_income(t) * self.comm_rate(t)

    with pytest.raises(BuildError, match="must be Count"):
        check(trace(WrongUnit))


def test_unused_variable_warns_and_is_left_out_of_generated_code() -> None:
    class Extra(TermAssurance):
        @var
        def unused(self, t: Int) -> Factor:
            return self.v(t) * 2.0

    with pytest.warns(BuildWarning, match="unused is read by no other variable"):
        compiled = build(Extra)
    assert "V_unused" not in compiled.source
    assert "V_reserve" in compiled.source


def test_init_value_must_have_the_declared_unit() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def y(self, t: Int) -> Count:
            return self.mp.pols_if_init * 1.0

        @var(init=lambda mp: mp.pols_if_init * mp.premium)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * 1.0

    _fails(Bad, "init value of x is Money", UnitError)


def test_boundary_values_read_fields_first_and_variables_second() -> None:
    """An init or terminal callable receives the boundary scope: a name is a model point field
    first and otherwise a variable, read as a scalar or at a fixed month (design section 5)."""
    g = check(trace(Margin))
    assert ir.render(_init(g, "csm")) == "csm_0(t)"
    assert ir.render(_init(g, "lc")) == "at(fcf, 0)"
    assert g.variables["csm"].unit == u.MONEY
    assert g.variables["fcf"].start == 0  # the terminal, and at(fcf, 0) from lc's init
    # boundary reads are edges: csm_0 is scheduled before csm, fcf before lc
    index = {name: i for i, comp in enumerate(g.components) for name in comp}
    assert index["csm_0"] < index["csm"]
    assert index["fcf"] < index["lc"]
    assert index["fcf"] < index["csm_0"]

    class FieldWins(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def premium(self) -> MoneyPerPolicy:
            return self.mp.premium * 2.0

        @var(init=lambda g: g.premium * g.pols_if_init)
        def x(self, t: Int) -> Money:
            return self.x(t - 1) * 0.9

    g2 = check(trace(FieldWins))
    assert ir.render(_init(g2, "x")) == "(mp.premium * mp.pols_if_init)"

    class Neither(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def csm_0(self) -> Money:
            return self.mp.premium * self.mp.pols_if_init

        @var(init=lambda g: g.csm_zero())
        def x(self, t: Int) -> Money:
            return self.x(t - 1) * 0.9

    exc = _fails(
        Neither,
        r"a boundary value reads 'csm_zero', which is neither a model point field \(k, "
        r"policy_id, pols_if_init, premium, rem_m\) nor a variable of the model \(csm_0, x\)",
    )
    loc = getattr(exc, "loc", None)
    assert loc is not None
    assert loc.function.endswith("Neither.<lambda>")


def test_boundary_values_reject_what_moves_with_time() -> None:
    """The checker allows fields, tables, run constants, scalar variables, ``at``/``sum_t``/
    ``scen.x.at`` with constant months, arithmetic, ``where`` and calls in a boundary; it rejects
    ``t``, relative reads of time variables and scenario inputs, and fixed reads whose month is
    not a constant. The message lists every fixed month a boundary may read, a stage's
    ``series.x.at(k)`` included."""
    fixed = (
        "a boundary value is fixed before the months run, from the model point, tables, run "
        r"constants, variables without a time dimension and fixed months \(at\(x, k\), "
        r"scen.x.at\(k\), series.x.at\(k\) with a constant k\)"
    )

    class PerPolicyMonth(Margin):
        @var(init=lambda g: at(g.fcf, g.k))
        def lc(self, t: Int) -> Money:
            return self.lc(t - 1) * 0.9

    _fails(PerPolicyMonth, rf"the init value of lc reads at\(fcf, mp.k\); {fixed}")

    class Allowed(Margin):
        @var(init=lambda g: where(g.csm_0() > 0, g.csm_0(), sum_t(g.cf) * 0.0))
        def csm(self, t: Int) -> Money:
            return self.csm(t - 1) * 0.95

    check(trace(Allowed))

    # the authoring API cannot write these into a boundary (no t, no scenario accessor there);
    # the checker still refuses them at the IR level
    for node, text in (
        (
            ir.Where(
                None,
                ir.Binary(None, "gt", ir.TimeIndex(None), ir.Const(None, 0)),
                ir.VarRead(None, "csm_0", 0),
                ir.Const(None, 0.0),
            ),
            "t",
        ),
        (ir.VarRead(None, "cf", -1), r"cf\(t-1\)"),
        (ir.VarRead(None, "fcf", 0), r"fcf\(t\)"),
        (ir.ScenAt(None, "cpi", ir.MpField(None, "k")), r"scen.cpi.at\(mp.k\)"),
    ):
        g = trace(Margin)
        g.scenarios["cpi"] = ir.ScenarioDecl("cpi", u.MONEY)
        g.variables["lc"].init = node
        with pytest.raises(BuildError, match=rf"the init value of lc reads {text}; {fixed}"):
            check(g)


def test_a_boundary_may_not_read_a_member_of_its_own_component() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=lambda g: at(g.y, 0))
        def x(self, t: Int) -> Money:
            return self.x(t - 1) * 0.9

        @var(domain="state")
        def y(self, t: Int) -> Money:
            return self.x(t) * 2.0

    exc = _fails(
        Bad,
        r"the init value of x reads at\(y, 0\), and y depends on x: compute the boundary from a "
        "variable outside the component",
    )
    loc = getattr(exc, "loc", None)
    assert loc is not None
    assert loc.function.endswith("Bad.<lambda>")


def test_a_backward_component_member_without_a_terminal_reaches_the_horizon() -> None:
    """The annuity's {bel, inv_exp}: bel reads inv_exp(t+1) and inv_exp reads bel(t). Only bel
    carries a terminal; inv_exp's formula reaches H and the backend evaluates it there before
    the loop (design section 4)."""
    g = check(trace(Expensed))
    assert ["bel", "inv_exp"] in g.components
    assert (g.variables["bel"].direction, g.variables["bel"].start) == ("backward", 0)
    assert (g.variables["inv_exp"].direction, g.variables["inv_exp"].start) == ("backward", 1)
    assert g.variables["inv_exp"].terminal is None
    assert g.variables["inv_exp"].formula_range == (1, "H")
    assert g.variables["bel"].formula_range == (0, "H-1")


def test_a_read_at_t_plus_one_from_a_formula_that_reaches_the_horizon_fails() -> None:
    """Without rule 7's terminal demand, _check_domains alone keeps a backward component sound:
    a member whose formula reaches H may not read t + 1, whatever it reads."""
    outside = (
        "{reader} reads {read}, which is outside the domain of {target} at t = H; a variable "
        "that reads a later month is backward recursive and declares a terminal value"
    )

    class StartOneReadsStartOne(Product):
        model_point = SIMPLE_SCHEMA

        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return (self.inv_exp(t + 1) * 0.9) + (self.bel(t + 1) * 0.9)

        @var
        def inv_exp(self, t: Int) -> Money:
            return (self.bel(t) * 0.01) + (self.fee(t + 1) * 0.5)

        @var
        def fee(self, t: Int) -> Money:
            return self.inv_exp(t) * 0.1

    _fails(
        StartOneReadsStartOne,
        re.escape(outside.format(reader="inv_exp", read="fee(t+1)", target="fee")),
    )

    class StateReadsTerminal(Product):
        model_point = SIMPLE_SCHEMA

        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return (self.inv_exp(t + 1) * 0.9) + (self.bel(t + 1) * 0.9)

        @var(domain="state")
        def inv_exp(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.01

    _fails(
        StateReadsTerminal,
        re.escape(outside.format(reader="inv_exp", read="bel(t+1)", target="bel")),
    )

    class StartOneReadsTerminal(Product):
        model_point = SIMPLE_SCHEMA

        @var(terminal=0)
        def bel(self, t: Int) -> Money:
            return (self.inv_exp(t + 1) * 0.9) + (self.bel(t + 1) * 0.9)

        @var
        def inv_exp(self, t: Int) -> Money:
            return self.bel(t + 1) * 0.01

    _fails(
        StartOneReadsTerminal,
        re.escape(outside.format(reader="inv_exp", read="bel(t+1)", target="bel")),
    )

    class NoTerminal(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def bel(self, t: Int) -> Money:
            return (self.inv_exp(t + 1) * 0.9) + (self.bel(t + 1) * 0.9)

        @var
        def inv_exp(self, t: Int) -> Money:
            return self.bel(t) * 0.01

    _fails(
        NoTerminal, re.escape(outside.format(reader="bel", read="inv_exp(t+1)", target="inv_exp"))
    )


def test_complete_reads_cannot_be_recursive() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * self.scale()

        @var
        def scale(self) -> Factor:
            return sum_t(self.x) / sum_t(self.x)

    _fails(Bad, "complete read cannot be part of a recursion")


def test_prob_tag_conversion_is_the_only_path() -> None:
    class Ok(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def q_m(self, t: Int) -> Prob[Monthly]:
            return to_monthly_prob(0.01)

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * (1 - self.q_m(t))

    check(trace(Ok))

    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def q_a(self, t: Int) -> Prob[Annual]:
            return 0.1 * 1.0  # type: ignore[return-value]

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * (1 - self.q_a(t))  # type: ignore[operator]

    _fails(Bad, "Prob\\[Annual\\]", UnitError)


def test_run_constants_are_scalar_ints() -> None:
    class Dated(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def cal_year(self, t: Int) -> Int:
            return (self.run.m_val + t) // 12

        @var
        def months(self) -> Int:
            return self.run.horizon

        @var
        def half(self) -> Factor:
            return self.run.horizon / 2

    g = check(trace(Dated))
    assert g.variables["months"].expr.unit == u.INT
    assert g.variables["cal_year"].unit == u.INT
    assert g.variables["months"].dims == frozenset()
    assert g.variables["half"].dims == frozenset()
    assert g.variables["cal_year"].dims == {"time"}
    assert g.variables["cal_year"].start == 1

    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def x(self) -> Money:
            return self.run.m_val * self.mp.premium  # type: ignore[operator, return-value]

    _fails(Bad, "unit mismatch: Int \\* MoneyPerPolicy", UnitError)


def test_model_point_needs_schema_and_unit_fields() -> None:
    class NoSchema(Product):
        @var
        def x(self, t: Int) -> Money:
            return self.mp.premium * 1.0

    with pytest.raises(BuildError, match="declares no model_point schema"):
        trace(NoSchema)


def test_money_per_policy_division_rule() -> None:
    class Ok(Product):
        model_point = SIMPLE_SCHEMA

        @var(init=lambda mp: mp.pols_if_init)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * 0.5

        @var
        def m(self, t: Int) -> Money:
            return self.x(t) * self.mp.premium

        @var
        def pp(self, t: Int) -> MoneyPerPolicy:
            return where(self.x(t) > 0, self.m(t) / self.x(t), 0)

    check(trace(Ok))


def test_schema_contracts_are_available_on_the_graph() -> None:
    g = trace(TermAssurance)
    assert set(g.tables) == {"mort_table", "lapse_table", "comm_table", "rates", "expenses"}
    assert g.tables[MORT_TABLE.name].keys[1].hi == 120
    assert g.tables[RATES.name].columns[0].unit.name == "Rate[Annual]"
    assert g.fields["sex"].enum == ("M", "F")
    assert g.fields["rem_m"].derived
    assert SCHEMA.horizon == "rem_m"
