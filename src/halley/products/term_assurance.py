"""Level term assurance: reference case 1.

The model of ``docs/spec/term-assurance.md`` (version 0.1), written from its sections 15.1 and
15.2 so that every variable name matches the spec and the spreadsheet column it reconciles to.
Also here: the model point schema (section 3) and the assumption table contracts (6.8). The
synthetic basis tables and the ten reference model points are built by
:mod:`halley.products.term_assurance_data`.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from halley.authoring import (
    Annual,
    Column,
    Count,
    Derived,
    Factor,
    Field,
    Fraction,
    Int,
    Key,
    LoadContext,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    Rule,
    Schema,
    Table,
    at,
    sum_t,
    to_monthly,
    to_monthly_prob,
    var,
    where,
)
from halley.authoring.dates import age_last_birthday, day, months_between

VALUATION_DATE = dt.date(2027, 1, 1)
"""The reference valuation date (section 13)."""

BASIS_VERSION = "term-ref-basis-0.1"

# ------------------------------------------------------------------- model point schema (3)


def _valuation(c: LoadContext, like: NDArray[Any]) -> NDArray[Any]:
    return np.broadcast_to(c.valuation, like.shape)


def _age_at_entry(c: LoadContext) -> NDArray[Any]:
    return age_last_birthday(c.birth_date, c.entry_date)


def _dur_m(c: LoadContext) -> NDArray[Any]:
    return months_between(c.entry_date, _valuation(c, c.entry_date))


def _term_m(c: LoadContext) -> NDArray[Any]:
    return months_between(c.entry_date, c.maturity_date)


def _rem_m(c: LoadContext) -> NDArray[Any]:
    return c.term_m - c.dur_m


def _prem_term_m(c: LoadContext) -> NDArray[Any]:
    return 12 * c.prem_term_y


def _prem_rem_m(c: LoadContext) -> NDArray[Any]:
    return np.maximum(c.prem_term_m - c.dur_m, 0)


def _dur_y_last(c: LoadContext) -> NDArray[Any]:
    return (c.term_m - 1) // 12


def _scale_money(c: LoadContext) -> NDArray[Any]:
    return c.sum_assured * c.pols_if_init


def _first_of_month(d: dt.date) -> str | None:
    return None if d.day == 1 else "must be the first day of a calendar month (spec 4.1)"


SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True, doc="Identifier, used only to label outputs"),
        Field("sex", "enum", enum=("M", "F"), doc="Mortality table selector"),
        Field("birth_date", "date", doc="Date of birth of the life insured"),
        Field("entry_date", "date", doc="Date the policy came into force"),
        Field("maturity_date", "date", doc="Date on which cover ends"),
        Field("prem_term_y", "int", "Int", ge=1, doc="Premium-paying term in whole years"),
        Field("sum_assured", "number", "MoneyPerPolicy", gt=0, le=10_000_000),
        Field("premium", "number", "MoneyPerPolicy", gt=0, le=100_000),
        Field("pols_if_init", "number", "Count", gt=0, le=1_000_000),
    ),
    derived=(
        Derived("age_at_entry", "Int", _age_at_entry, ge=18, le=65),
        Derived("dur_m", "Int", _dur_m, ge=0, doc="entry on or before the valuation date"),
        Derived("term_m", "Int", _term_m, ge=12, le=480),
        Derived("rem_m", "Int", _rem_m, ge=1, le=480, doc="at least one month of cover left"),
        Derived("prem_term_m", "Int", _prem_term_m, ge=12),
        Derived("prem_rem_m", "Int", _prem_rem_m, ge=0),
        Derived("dur_y_last", "Int", _dur_y_last, ge=0, le=39),
        Derived("scale_money", "Money", _scale_money, kind="number", ge=0),
    ),
    rules=(
        Rule(
            "maturity_date.day = entry_date.day",
            lambda c: day(c.maturity_date) == day(c.entry_date),
            "the term is a whole number of months (spec 3.1)",
        ),
        Rule("prem_term_m <= term_m", lambda c: c.prem_term_m <= c.term_m),
    ),
    horizon="rem_m",
    valuation_date_rule=_first_of_month,
)

# ------------------------------------------------------------- assumption table contracts (6.8)

MORT_TABLE = Table(
    "mort_table",
    keys=(Key("sex", enum=("M", "F")), Key("age", lo=18, hi=120)),
    columns=(Column("q_ann", "Prob[Annual]"),),
    doc="Annual mortality by sex and age last birthday",
)
LAPSE_TABLE = Table(
    "lapse_table",
    keys=(Key("dur_y", lo=0, hi=5),),
    columns=(Column("w_ann", "Prob[Annual]"),),
    doc="Annual lapse rate by completed policy years, clamped at 5",
)
COMM_TABLE = Table(
    "comm_table",
    keys=(Key("dur_y", lo=0, hi=1),),
    columns=(Column("comm_rate", "Fraction", ge=0, le=1),),
    doc="Commission as a proportion of premium, by completed policy years, clamped at 1",
)
RATES = Table(
    "rates",
    keys=(Key("name", enum=("i", "e")),),
    columns=(Column("rate", "Rate[Annual]"),),
    doc="Annual effective interest (i) and expense inflation (e)",
)
EXPENSES = Table(
    "expenses",
    keys=(Key("name", enum=("maint_base", "claim_exp_base")),),
    columns=(Column("amount", "MoneyPerPolicy", ge=0),),
    doc="Maintenance expense per policy per month and claim expense per death, time-0 prices",
)


# --------------------------------------------------------------------------- the model (15.2)


class TermAssurance(Product):
    """Level term assurance, spec version 0.1. No build-time switches (section 2)."""

    model_point = SCHEMA
    assumption_tables = (MORT_TABLE, LAPSE_TABLE, COMM_TABLE, RATES, EXPENSES)
    outputs = (
        "deaths",
        "lapses",
        "maturities",
        "pols_if",
        "commission_pp",
        "premium_income",
        "commission",
        "maint_expense",
        "death_outgo",
        "claim_expense",
        "net_cf",
        "pv_net_cf_t",
        "reserve",
        "reserve_pp",
        "reserve_floored",
        "pv_premiums",
        "pv_commission",
        "pv_maint_expense",
        "pv_death_outgo",
        "pv_claim_expense",
        "pv_net_cf",
        "reserve_0",
        "reserve_pp_0",
        "reserve_floored_0",
        "maturities_n",
    )
    tolerance_scales: ClassVar[dict[str, str]] = {
        "Money": "scale_money",
        "MoneyPerPolicy": "sum_assured",
        "Count": "pols_if_init",
    }

    # ---- basis constants (section 6) --------------------------------------------------
    @var
    def i(self) -> Rate[Annual]:
        return self.tables.rates["i"]

    @var
    def e(self) -> Rate[Annual]:
        return self.tables.rates["e"]

    @var
    def i_m(self) -> Rate[Monthly]:
        return to_monthly(self.i())  # (1 + i) ** (1/12) - 1

    @var
    def v_m(self) -> Factor:
        return (1 + self.i()) ** (-1 / 12)  # a direct power, not 1 / (1 + i_m)

    @var
    def e_m(self) -> Rate[Monthly]:
        return to_monthly(self.e())

    # ---- age and lookup keys (section 5) ----------------------------------------------
    @var
    def dur_y(self, t: Int) -> Int:
        k = (self.mp.dur_m + t - 1) // 12  # completed policy years at the start of month t
        return where(k < self.mp.dur_y_last, k, self.mp.dur_y_last)

    @var
    def age(self, t: Int) -> Int:
        return self.mp.age_at_entry + self.dur_y(t)

    # ---- rates (section 6) -----------------------------------------------------------
    @var
    def q_ann(self, t: Int) -> Prob[Annual]:
        return self.tables.mort_table[self.mp.sex, self.age(t)]

    @var
    def q_m(self, t: Int) -> Prob[Monthly]:
        return where(t <= self.mp.rem_m, to_monthly_prob(self.q_ann(t)), 0)

    @var
    def w_ann(self, t: Int) -> Prob[Annual]:
        key = where(self.dur_y(t) < 5, self.dur_y(t), 5)
        return where(t <= self.mp.prem_rem_m, self.tables.lapse_table[key], 0)

    @var
    def w_m(self, t: Int) -> Prob[Monthly]:
        return to_monthly_prob(self.w_ann(t))

    @var
    def comm_rate(self, t: Int) -> Fraction:
        return self.tables.comm_table[where(self.dur_y(t) < 1, self.dur_y(t), 1)]

    @var(domain="state")
    def infl(self, t: Int) -> Factor:
        return (1 + self.e_m()) ** t  # infl(0) = 1

    # ---- decrements (section 7) ------------------------------------------------------
    @var(role="decrement")
    def deaths(self, t: Int) -> Count:
        return self.pols_if(t - 1) * self.q_m(t)

    @var(role="decrement")
    def lapses(self, t: Int) -> Count:
        return (self.pols_if(t - 1) - self.deaths(t)) * self.w_m(t)

    @var(role="decrement")
    def maturities(self, t: Int) -> Count:
        return where(t == self.mp.rem_m, self.pols_if(t - 1) - self.deaths(t) - self.lapses(t), 0)

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def pols_if(self, t: Int) -> Count:
        return self.pols_if(t - 1) - self.deaths(t) - self.lapses(t) - self.maturities(t)

    # ---- per-policy amounts (8.1) ----------------------------------------------------
    @var
    def premium_pp(self, t: Int) -> MoneyPerPolicy:
        return where(t <= self.mp.prem_rem_m, self.mp.premium, 0)

    @var
    def commission_pp(self, t: Int) -> MoneyPerPolicy:
        return self.premium_pp(t) * self.comm_rate(t)

    @var
    def maint_pp(self, t: Int) -> MoneyPerPolicy:
        return self.tables.expenses["maint_base"] * self.infl(t)

    @var
    def claim_exp_pp(self, t: Int) -> MoneyPerPolicy:
        return self.tables.expenses["claim_exp_base"] * self.infl(t)

    # ---- in-force cash flows (8.2) ---------------------------------------------------
    @var(timing="start")
    def premium_income(self, t: Int) -> Money:
        return self.pols_if(t - 1) * self.premium_pp(t)

    @var(timing="start")
    def commission(self, t: Int) -> Money:
        return self.premium_income(t) * self.comm_rate(t)

    @var(timing="end")
    def maint_expense(self, t: Int) -> Money:
        return self.pols_if(t - 1) * self.maint_pp(t)

    @var(timing="end")
    def death_outgo(self, t: Int) -> Money:
        return self.deaths(t) * self.mp.sum_assured

    @var(timing="end")
    def claim_expense(self, t: Int) -> Money:
        return self.deaths(t) * self.claim_exp_pp(t)

    # ---- aggregates (8.3) ------------------------------------------------------------
    @var
    def net_cf(self, t: Int) -> Money:
        return (
            self.premium_income(t)
            - self.commission(t)
            - self.maint_expense(t)
            - self.death_outgo(t)
            - self.claim_expense(t)
        )

    @var(timing="end")
    def net_outgo(self, t: Int) -> Money:
        return (
            self.death_outgo(t)
            + self.claim_expense(t)
            + self.maint_expense(t)
            - (self.premium_income(t) - self.commission(t)) * (1 + self.i_m())
        )

    # ---- discounting and present values (section 9) ----------------------------------
    @var
    def v(self, t: Int) -> Factor:
        return self.v_m()

    @var(domain="state")
    def disc(self, t: Int) -> Factor:
        return self.v_m() ** t  # disc(0) = 1

    @var
    def pv_net_cf_t(self, t: Int) -> Money:
        return (self.premium_income(t) - self.commission(t)) * self.disc(t - 1) - (
            self.maint_expense(t) + self.death_outgo(t) + self.claim_expense(t)
        ) * self.disc(t)

    @var
    def pv_premiums_t(self, t: Int) -> Money:
        return self.premium_income(t) * self.disc(t - 1)

    @var
    def pv_commission_t(self, t: Int) -> Money:
        return self.commission(t) * self.disc(t - 1)

    @var
    def pv_maint_expense_t(self, t: Int) -> Money:
        return self.maint_expense(t) * self.disc(t)

    @var
    def pv_death_outgo_t(self, t: Int) -> Money:
        return self.death_outgo(t) * self.disc(t)

    @var
    def pv_claim_expense_t(self, t: Int) -> Money:
        return self.claim_expense(t) * self.disc(t)

    # ---- reserve (section 10) --------------------------------------------------------
    @var(terminal=0, role="reserve")
    def reserve(self, t: Int) -> Money:
        return (self.reserve(t + 1) + self.net_outgo(t + 1)) * self.v(t + 1)

    @var
    def reserve_pp(self, t: Int) -> MoneyPerPolicy:
        return where(self.pols_if(t) > 0, self.reserve(t) / self.pols_if(t), 0)

    @var
    def reserve_floored(self, t: Int) -> Money:
        return where(self.reserve(t) > 0, self.reserve(t), 0)

    # ---- per-policy scalars (11.2): fixed-order sums over t ---------------------------
    @var
    def pv_premiums(self) -> Money:
        return sum_t(self.pv_premiums_t)

    @var
    def pv_commission(self) -> Money:
        return sum_t(self.pv_commission_t)

    @var
    def pv_maint_expense(self) -> Money:
        return sum_t(self.pv_maint_expense_t)

    @var
    def pv_death_outgo(self) -> Money:
        return sum_t(self.pv_death_outgo_t)

    @var
    def pv_claim_expense(self) -> Money:
        return sum_t(self.pv_claim_expense_t)

    @var
    def pv_net_cf(self) -> Money:
        return (
            self.pv_premiums()
            - self.pv_commission()
            - self.pv_maint_expense()
            - self.pv_death_outgo()
            - self.pv_claim_expense()
        )

    @var
    def reserve_0(self) -> Money:
        return at(self.reserve, 0)

    @var
    def reserve_pp_0(self) -> MoneyPerPolicy:
        return at(self.reserve_pp, 0)

    @var
    def reserve_floored_0(self) -> Money:
        return at(self.reserve_floored, 0)

    @var
    def maturities_n(self) -> Count:
        return at(self.maturities, self.mp.rem_m)
