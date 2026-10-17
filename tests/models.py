# pyright: basic
"""Small models and inputs the engine tests share."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pyarrow as pa

from halley import (
    Annual,
    Column,
    Count,
    Factor,
    Field,
    InputRule,
    Int,
    Key,
    Money,
    MoneyPerPolicy,
    Monthly,
    Prob,
    Product,
    Rate,
    RunResult,
    RunSettings,
    ScenarioInput,
    ScenarioRule,
    Schema,
    Table,
    TableRule,
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
    before the loop steps both members back together. The reserve's terminal is a year's premium
    per life, not zero, so that ``inv_exp(H)`` is non-zero and ``bel(H - 1)`` depends on the
    pre-loop having run."""

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

    @var(terminal=lambda mp: mp.pols_if_init * mp.premium)
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
    name: str,
    u_lo: int,
    u_hi: int,
    n_scen: int,
    fn: Callable[[int, int], float],
    month_column: str = "t",
) -> Any:
    """A long-format scenario file: ``scenario``, the month column (``t`` or ``u``) and the
    value column ``name``, covering months ``u_lo..u_hi`` for every scenario."""
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
            month_column: pa.array(rows_u, pa.int32()),
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
    """What ``Expensed`` computes, written independently with plain floats: the reserve ends at
    a year's premium per life, the investment expense at ``H`` is its formula on that terminal,
    then both recursions step back, the reserve first because the expense of a month is charged
    on that month's reserve."""
    v = 1.04 ** (-1 / 12)
    ie = ((1 + 0.012) ** (1 / 12)) - 1
    x = np.array([x0 * 0.99**t for t in range(horizon + 1)])
    cf = np.zeros(horizon + 1)
    for t in range(1, horizon + 1):
        cf[t] = x[t - 1] * premium
    bel = np.zeros(horizon + 1)
    inv_exp = np.zeros(horizon + 1)
    bel[horizon] = x0 * premium
    inv_exp[horizon] = ie * bel[horizon]
    for t in range(horizon - 1, -1, -1):
        bel[t] = (cf[t + 1] + inv_exp[t + 1] * v) + v * bel[t + 1]
        if t >= 1:
            inv_exp[t] = ie * bel[t]
    return {"x": x, "cf": cf, "bel": bel, "inv_exp": inv_exp}


# ------------------------------------------------------------------- input contracts (A3)

MORT_RULED = Table(
    "mort",
    keys=(Key("sex", enum=("M", "F")), Key("age", lo=20, hi=25)),
    columns=(Column("q", "Prob[Annual]"),),
    doc="A mortality table that must close at its last age",
    metadata=("base_year",),
    rules=(
        TableRule("q(., 25) = 1", lambda v: v.at("q", age=25) == 1.0, "the table closes at 25"),
    ),
)
PARAMS = Table(
    "params",
    keys=(),
    columns=(Column("alpha", "Factor", ge=0.5, le=1), Column("z_alpha", "Factor")),
    rules=(
        TableRule(
            "z_alpha = inv_cdf(alpha)",
            lambda v: abs(float(v.z_alpha) - NormalDist().inv_cdf(float(v.alpha))) <= 1e-12,
        ),
    ),
)
DF_INPUT = ScenarioInput(
    "df",
    "Factor",
    gt=0,
    le=2,
    column="rf_df",
    covers=(0, "H"),
    rules=(
        ScenarioRule("df(s, 0) = 1", lambda v: v.at(0) == 1.0, "a discount factor starts at 1"),
    ),
)
CPI_INPUT = ScenarioInput(
    "cpi",
    "Index",
    gt=0,
    covers=(-14, "H"),
    rules=(
        ScenarioRule(
            "history identical across scenarios",
            lambda v: np.where(v.months[:, None] <= 0, v.values == v.values[:, :1], True),
        ),
    ),
)


class Ruled(Product):
    """Tables with required metadata and rules (one keyed, one keyless), scenario inputs with a
    renamed value column, bounds, coverage beyond the months read and rules, and input rules
    across all of them."""

    model_point = SIMPLE_SCHEMA
    assumption_tables = (MORT_RULED, PARAMS)
    scenario_inputs = (DF_INPUT, CPI_INPUT)
    input_rules = (
        InputRule(
            "base_year is the valuation year",
            lambda v: int(v.tables["mort"].metadata["base_year"]) == v.valuation_date.year,
        ),
        InputRule(
            "q * alpha <= 1 on the grid",
            lambda v: v.tables["mort"].q * v.tables["params"].alpha <= 1,
        ),
        InputRule(
            "the cpi file reaches the horizon",
            lambda v: v.scenarios["cpi"].u_max >= v.horizon,
        ),
        InputRule(
            "every k is at most the horizon",
            lambda v: v.model_points.k <= v.horizon,
            "a fixed month read at k",
        ),
        InputRule(
            "the valuation month is the run constant",
            lambda v: v.valuation_month == 12 * v.valuation_date.year + v.valuation_date.month - 1,
        ),
    )

    @var
    def q(self, t: Int) -> Prob[Annual]:
        return self.tables.mort["M", where(t < 5, 20 + t, 25)]

    @var
    def alpha(self) -> Factor:
        return self.tables.params.alpha

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return self.scen.df(t)

    @var
    def esc(self, t: Int) -> Factor:
        return self.scen.cpi(t - 3) / self.scen.cpi(t - 15)

    @var
    def x(self, t: Int) -> Money:
        return self.mp.premium * self.mp.pols_if_init * self.df(t) * self.esc(t) * self.alpha()


def ruled_tables(
    base_year: str | None = "2027", q_last: float = 1.0, z_alpha: float | None = None
) -> dict[str, Any]:
    """The tables ``Ruled`` reads, with ``halley.*`` metadata on the mortality table and a bare
    ``version`` on the parameter table."""
    rows = [
        {"sex": s, "age": a, "q": q_last if a == 25 else 0.01 * (a - 19)}
        for s in ("M", "F")
        for a in range(20, 26)
    ]
    meta = {b"halley.version": b"ruled-0.1"}
    if base_year is not None:
        meta[b"halley.base_year"] = base_year.encode()
    mort = pa.Table.from_pylist(rows).replace_schema_metadata(meta)
    z = NormalDist().inv_cdf(0.75) if z_alpha is None else z_alpha
    params = pa.table(
        {"alpha": pa.array([0.75], pa.float64()), "z_alpha": pa.array([z], pa.float64())}
    ).replace_schema_metadata({b"version": b"params-7"})
    return {"mort": mort, "params": params}


def ruled_scenarios(
    horizon: int = 12,
    n_scen: int = 2,
    df0: float = 1.0,
    cpi_lo: int = -14,
    cpi_hi: int | None = None,
    history_shift: float = 0.0,
) -> dict[str, Any]:
    """The scenario files ``Ruled`` reads: the discount file with a ``t`` month column and the
    value column ``rf_df``, the CPI file with a ``u`` month column."""
    df = scenario_file(
        "rf_df", 0, horizon, n_scen, lambda s, u: df0 if u == 0 else (0.99 - 0.01 * s) ** u
    )

    def cpi_at(s: int, u: int) -> float:
        if u > 0:
            return 100.0 * (1.002 + 0.001 * s) ** u
        return 100.0 * 1.002**u + history_shift * s

    cpi = scenario_file(
        "cpi", cpi_lo, horizon if cpi_hi is None else cpi_hi, n_scen, cpi_at, month_column="u"
    )
    return {"df": df, "cpi": cpi}


def run_ruled(
    tmp_path: Path,
    tables: dict[str, Any] | None = None,
    scenarios: dict[str, Any] | None = None,
    **settings: Any,
) -> RunResult:
    return run(
        build(Ruled),
        model_points=simple_model_points(),
        tables=ruled_tables() if tables is None else tables,
        scenarios=ruled_scenarios() if scenarios is None else scenarios,
        settings=RunSettings(valuation_date=VAL, **settings),
        run_dir=tmp_path / "ruled",
    )


GROUP_SCHEMA = Schema(
    fields=(
        Field("group_id", "string", unique=True),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
    id_field="group_id",
)


class Grouped(Product):
    """A schema whose id field is not ``policy_id``: the outputs are keyed on it."""

    model_point = GROUP_SCHEMA

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.9

    @var
    def total(self) -> Count:
        return sum_t(self.x)


def group_model_points(n: int = 2, rem_m: int = 6) -> Any:
    return pa.table(
        {
            "group_id": pa.array([f"G{i}" for i in range(n)], pa.string()),
            "pols_if_init": pa.array([10.0 * (i + 1) for i in range(n)], pa.float64()),
            "rem_m": pa.array([rem_m] * n, pa.int32()),
        }
    )


# ------------------------------------------------------------ a non-literal exponent (B0r)

POWER_DEN = 100_000
"""The denominator of ``Powered.x``: month values ``(t + j) / POWER_DEN``."""

POWER_SCHEMA = Schema(
    fields=(
        Field("policy_id", "string", unique=True),
        int_field("k", ge=-1, le=3),
        int_field("j", ge=1, le=100 * POWER_DEN),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)


class Powered(Product):
    """A model point field as an exponent, ``x(t) ** self.mp.k``, outside a recursive component
    (``whole``, emitted ``V_x[1:H+1] ** M_k``) and inside one (``term``, emitted
    ``V_x[t] ** m_k``, summed by ``acc``). Whether NumPy takes its reciprocal or square fast
    path for k = -1 or 2 there depends on the shapes of the batch, not on the formula."""

    model_point = POWER_SCHEMA

    @var
    def x(self, t: Int) -> Factor:
        return (t + self.mp.j) / POWER_DEN

    @var
    def whole(self, t: Int) -> Factor:
        return self.x(t) ** self.mp.k

    @var(init=0.0)
    def acc(self, t: Int) -> Factor:
        return self.acc(t - 1) + self.term(t)

    @var
    def term(self, t: Int) -> Factor:
        # a member of acc's component: the power is evaluated month by month
        return where(self.acc(t - 1) >= 0, self.x(t) ** self.mp.k, 0.0)


def powered_model_points(js: list[int], ks: list[int], rem_m: int = 12) -> Any:
    """``POWER_SCHEMA`` model points ``K0, K1, ...``, one per offset ``j`` and exponent ``k``."""
    n = len(js)
    return pa.table(
        {
            "policy_id": pa.array([f"K{i}" for i in range(n)], pa.string()),
            "k": pa.array(ks, pa.int32()),
            "j": pa.array(js, pa.int32()),
            "rem_m": pa.array([rem_m] * n, pa.int32()),
        }
    )
