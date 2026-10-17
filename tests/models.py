# pyright: basic
"""Small models and inputs the engine tests share."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from halley import (
    Column,
    Count,
    Factor,
    Field,
    Int,
    Money,
    MoneyPerPolicy,
    Monthly,
    Product,
    Rate,
    RunResult,
    RunSettings,
    ScenarioInput,
    Schema,
    Table,
    at,
    build,
    pv_month,
    run,
    sum_t,
    to_monthly,
    var,
    where,
)
from halley.authoring import int_field

VAL = dt.date(2027, 1, 1)

SIMPLE_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        int_field("k", ge=0, le=1000),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


def simple_model_points(n: int = 3, rem_m: int = 12) -> Any:
    ids = [f"S{i:02d}" for i in range(n)]
    return pa.table(
        {
            "policy_id": pa.array(ids, pa.string()),
            "k": pa.array([2 + i for i in range(n)], pa.int32()),
            "premium": pa.array([10.0 * (i + 1) for i in range(n)], pa.float64()),
            "pols_if_init": pa.array([1.0 + i for i in range(n)], pa.float64()),
            "rem_m": pa.array([rem_m] * n, pa.int32()),
        }
    )


class Growth(Product):
    """A forward recursion, a backward recursion, a masked division, a sum and fixed reads."""

    model_point = SIMPLE_SCHEMA

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.99

    @var
    def cf(self, t: Int) -> Money:
        return self.x(t - 1) * self.mp.premium

    @var(terminal=0)
    def r(self, t: Int) -> Money:
        return (self.r(t + 1) + self.cf(t + 1)) * 0.5

    @var
    def ratio(self, t: Int) -> MoneyPerPolicy:
        return where(self.x(t) > 0, self.r(t) / self.x(t), 0)

    @var
    def masked(self, t: Int) -> Factor:
        return where(t > 2, 1.0 / (t - 2), 0.0)

    @var
    def total(self) -> Money:
        return sum_t(self.cf)

    @var
    def r0(self) -> Money:
        return at(self.r, 0)

    @var
    def cf_at_k(self) -> Money:
        return at(self.cf, self.mp.k)


class Indexed(Product):
    """Scenario inputs: a discount curve read at t and t-1, a CPI path read three months back
    and at month 0, so the loader must find history rows."""

    model_point = SIMPLE_SCHEMA
    scenario_inputs = (ScenarioInput("df", "Factor"), ScenarioInput("cpi", "Factor"))

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return self.scen.df(t)

    @var
    def v(self, t: Int) -> Factor:
        return self.df(t) / self.df(t - 1)

    @var
    def esc(self, t: Int) -> Factor:
        return self.scen.cpi(t - 3) / self.scen.cpi.at(0)

    @var(init=lambda mp: mp.premium, timing="start")
    def ann_pp(self, t: Int) -> MoneyPerPolicy:
        return self.ann_pp(t - 1) * self.esc(t)

    @var
    def pay(self, t: Int) -> Money:
        return self.mp.pols_if_init * self.ann_pp(t)


class Expensed(Product):
    """A backward component of two members, the annuity's ``{bel, inv_exp}``: the reserve reads
    next month's investment expense, which is charged on the closing reserve of its own month.
    ``inv_exp`` carries no terminal; its formula reaches ``H``, where the backend evaluates it
    before the loop steps both members back together."""

    model_point = SIMPLE_SCHEMA

    @var(init=lambda mp: mp.pols_if_init)
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.99

    @var(timing="start")
    def cf(self, t: Int) -> Money:
        return self.x(t - 1) * self.mp.premium

    @var
    def v_m(self) -> Factor:
        return 1.04 ** (-1 / 12)

    @var
    def v(self, t: Int) -> Factor:
        return self.v_m()

    @var
    def ie_m(self) -> Rate[Monthly]:
        return to_monthly(0.012)

    @var(terminal=0)
    def bel(self, t: Int) -> Money:
        return (pv_month(self.cf, t + 1, self.v) + pv_month(self.inv_exp, t + 1, self.v)) + (
            self.v(t + 1) * self.bel(t + 1)
        )

    @var(timing="end")
    def inv_exp(self, t: Int) -> Money:
        return self.ie_m() * self.bel(t)


MARGIN = Table("margin", keys=(), columns=(Column("share", "Factor", ge=0, le=1),))


class Margin(Product):
    """Boundary values computed from other variables (the M7 shape): ``csm`` starts at
    ``csm_0``, a scalar formed from the premium, a fixed month of the backward ``fcf`` and a
    table value; ``lc`` starts at ``at(fcf, 0)``, a backward variable outside its component."""

    model_point = SIMPLE_SCHEMA
    assumption_tables = (MARGIN,)

    @var(init=lambda mp: mp.pols_if_init)
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.99

    @var
    def cf(self, t: Int) -> Money:
        return self.x(t - 1) * self.mp.premium

    @var(terminal=0)
    def fcf(self, t: Int) -> Money:
        return (self.fcf(t + 1) + self.cf(t + 1)) * 0.9

    @var
    def premium_total(self) -> Money:
        return self.mp.pols_if_init * self.mp.premium

    @var
    def csm_0(self) -> Money:
        return (self.premium_total() - at(self.fcf, 0)) * self.tables.margin.share

    @var(init=lambda g: g.csm_0())
    def csm(self, t: Int) -> Money:
        return self.csm(t - 1) * 0.95

    @var(init=lambda g: at(g.fcf, 0))
    def lc(self, t: Int) -> Money:
        return self.lc(t - 1) * 0.9


def margin_table(share: float = 0.5) -> Any:
    return pa.table({"share": pa.array([share], pa.float64())})


def scenario_file(
    name: str, u_lo: int, u_hi: int, n_scen: int, fn: Callable[[int, int], float]
) -> Any:
    rows_s: list[int] = []
    rows_u: list[int] = []
    rows_x: list[float] = []
    for s in range(n_scen):
        for u in range(u_lo, u_hi + 1):
            rows_s.append(s)
            rows_u.append(u)
            rows_x.append(float(fn(s, u)))
    return pa.table(
        {
            "scenario": pa.array(rows_s, pa.int32()),
            "t": pa.array(rows_u, pa.int32()),
            name: pa.array(rows_x, pa.float64()),
        }
    )


def run_growth(tmp_path: Path, **settings: Any) -> RunResult:
    compiled = build(Growth)
    return run(
        compiled,
        model_points=simple_model_points(),
        settings=RunSettings(valuation_date=VAL, **settings),
        run_dir=tmp_path / "growth",
    )


def closed_form_growth(x0: float, premium: float, horizon: int) -> dict[str, np.ndarray]:
    """What ``Growth`` computes, written independently with plain floats."""
    x = np.array([x0 * 0.99**t for t in range(horizon + 1)])
    cf = np.zeros(horizon + 1)
    for t in range(1, horizon + 1):
        cf[t] = x[t - 1] * premium
    r = np.zeros(horizon + 1)
    for t in range(horizon - 1, -1, -1):
        r[t] = (r[t + 1] + cf[t + 1]) * 0.5
    return {"x": x, "cf": cf, "r": r}


def closed_form_expensed(x0: float, premium: float, horizon: int) -> dict[str, np.ndarray]:
    """What ``Expensed`` computes, written independently with plain floats: the investment
    expense at ``H`` is its formula on the terminal reserve, then both recursions step back,
    the reserve first because the expense of a month is charged on that month's reserve."""
    v = 1.04 ** (-1 / 12)
    ie = ((1 + 0.012) ** (1 / 12)) - 1
    x = np.array([x0 * 0.99**t for t in range(horizon + 1)])
    cf = np.zeros(horizon + 1)
    for t in range(1, horizon + 1):
        cf[t] = x[t - 1] * premium
    bel = np.zeros(horizon + 1)
    inv_exp = np.zeros(horizon + 1)
    inv_exp[horizon] = ie * bel[horizon]
    for t in range(horizon - 1, -1, -1):
        bel[t] = (cf[t + 1] + inv_exp[t + 1] * v) + v * bel[t + 1]
        if t >= 1:
            inv_exp[t] = ie * bel[t]
    return {"x": x, "cf": cf, "bel": bel, "inv_exp": inv_exp}
