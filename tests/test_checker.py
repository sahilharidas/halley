"""Checker rules, each with the deliberate mistake the term assurance spec says must fail
(section 15.3) and a few more. Every error must point at the model source."""

from __future__ import annotations

import pytest

from halley import (
    Annual,
    BuildError,
    BuildWarning,
    Count,
    Factor,
    Int,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    UnitError,
    build,
    check,
    sum_t,
    to_monthly_prob,
    trace,
    var,
    where,
)
from halley.products.term_assurance import MORT_TABLE, RATES, SCHEMA, TermAssurance
from models import SIMPLE_SCHEMA


def _fails(model: type[Product], match: str, error: type[Exception] = BuildError) -> Exception:
    with pytest.raises(error, match=match) as info:
        check(trace(model))
    exc = info.value
    loc = getattr(exc, "loc", None)
    assert loc is not None, "every checker error carries a model source location"
    assert loc.file.endswith(("test_checker.py", "term_assurance.py"))
    return exc


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


def test_init_may_not_depend_on_time_or_other_variables() -> None:
    class Bad(Product):
        model_point = SIMPLE_SCHEMA

        @var
        def y(self, t: Int) -> Count:
            return self.mp.pols_if_init * 1.0

        @var(init=lambda mp: mp.pols_if_init * mp.premium)
        def x(self, t: Int) -> Count:
            return self.x(t - 1) * 1.0

    _fails(Bad, "init value of x is Money", UnitError)


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
