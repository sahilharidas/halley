# pyright: basic
"""Small models and inputs the engine tests share."""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pyarrow as pa

from halley import (
    Aggregation,
    Annual,
    Basis,
    Column,
    CompiledModel,
    Count,
    CoverageUnits,
    Factor,
    Field,
    GroupSum,
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
    as_cu,
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
from halley.authoring.contracts import PROFIT_BUCKETS, group_schema

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


class Inverted(Growth):
    """``~`` of comparisons between plain numbers: of ``t`` and of a run constant inside a
    forward recursion and of a run constant in a variable without a time dimension, where the
    NumPy backend's generated code holds Python bools (``~True`` is -2, which is true), and of
    ``t`` in whole mode, where it holds an array. ``stepped`` and ``whole`` are 1.0 up to month
    9 and 0.0 after it; ``short`` is 0.0 over a horizon of more than 5 months."""

    @var(init=0.0)
    def stepped(self, t: Int) -> Factor:
        return self.stepped(t - 1) * 0 + where(~(t > 9) & ~(self.run.horizon < t), 1.0, 0.0)

    @var
    def whole(self, t: Int) -> Factor:
        return where(~(t > 9), 1.0, 0.0)  # type: ignore[return-value] - literals take the unit

    @var
    def short(self) -> Factor:
        return where(~(self.run.horizon > 5), 1.0, 0.0)  # type: ignore[return-value]


class IntPowers(Product):
    """An integer literal raised to an Int power, which the checker makes a Factor: over the
    month column (whole mode), inside a forward recursion (where ``t`` is a Python int), to a
    negative exponent, from a negative base, from a base built of integer literals alone in both
    modes, and in a variable without a time dimension. Over 70 months ``2 ** t`` and ``3 ** t``
    pass 2**63, where NumPy's int64 power wraps without a word."""

    model_point = SIMPLE_SCHEMA

    @var
    def whole(self, t: Int) -> Factor:
        return 2**t  # type: ignore[operator] - an int literal to an Int power

    @var(init=1.0)
    def stepped(self, t: Int) -> Factor:
        return self.stepped(t - 1) * 0 + 2**t  # type: ignore[operator]

    @var
    def halves(self, t: Int) -> Factor:
        return 2 ** (-t)  # type: ignore[operator]

    @var
    def signs(self, t: Int) -> Factor:
        return (-2) ** t  # type: ignore[operator]

    @var
    def chosen(self, t: Int) -> Factor:
        return where(t > 3, 3, 2) ** t  # type: ignore[operator]

    @var(init=1.0)
    def chosen_stepped(self, t: Int) -> Factor:
        return self.chosen_stepped(t - 1) * 0 + where(t > 3, 3, 2) ** t  # type: ignore[operator]

    @var
    def scalar(self) -> Factor:
        return 2**self.run.horizon  # type: ignore[operator]


def int_powers_expected(name: str, t: int) -> float:
    """What :class:`IntPowers` stores for ``name`` at month ``t``: the float power."""
    if name in ("whole", "stepped"):
        return 2.0**t
    if name == "halves":
        return 2.0**-t
    if name == "signs":
        return (-2.0) ** t
    return (3.0 if t > 3 else 2.0) ** t


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


# ------------------------------------------------------------ the aggregation stage (B1)


class Covered(Product):
    """A projection whose outputs a stage sums: coverage units (a month quantity), fulfilment
    cash flows (a backward state, so its domain starts at t = 0), a small expense and the
    premium (a scalar). A scenario input scales the claims, so a run may carry several
    scenarios."""

    model_point = SIMPLE_SCHEMA
    scenario_inputs = (ScenarioInput("shock", "Factor"),)
    outputs = ("x", "cf", "cu", "fcf", "fee", "premium_total")

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.98

    @var
    def cf(self, t: Int) -> Money:
        return (self.x(t - 1) * self.mp.premium) * self.scen.shock(t)

    @var
    def cu(self, t: Int) -> CoverageUnits:
        return as_cu(self.x(t - 1) * self.mp.premium)

    @var(terminal=0)
    def fcf(self, t: Int) -> Money:
        return (self.fcf(t + 1) + self.cf(t + 1)) * 0.995

    @var
    def fee(self, t: Int) -> Money:
        """Small beside fcf and cf, so that how three group totals are grouped shows."""
        return (self.x(t - 1) * self.mp.premium) * 0.0137

    @var
    def premium_total(self) -> Money:
        return (self.mp.pols_if_init * self.mp.premium) * 14.0


def covered_points(
    prefix: str, n: int = 3, rem_m: int = 12, scale: float = 1.0, ids: list[str] | None = None
) -> Any:
    """``SIMPLE_SCHEMA`` model points ``<prefix>00, <prefix>01, ...`` (or ``ids``) for
    ``Covered``."""
    names = ids if ids is not None else [f"{prefix}{i:02d}" for i in range(n)]
    n = len(names)
    return pa.table(
        {
            "policy_id": pa.array(names, pa.string()),
            "k": pa.array([2 + i for i in range(n)], pa.int32()),
            "premium": pa.array([scale * (10.0 + 3.0 * i) for i in range(n)], pa.float64()),
            "pols_if_init": pa.array([1.0 + 0.5 * i for i in range(n)], pa.float64()),
            "rem_m": pa.array([rem_m] * n, pa.int32()),
        }
    )


@functools.cache
def compiled_covered() -> CompiledModel:
    return build(Covered)


def run_covered(
    tmp_path: Path,
    name: str,
    points: Any,
    shocks: tuple[float, ...] = (1.0,),
    valuation_date: dt.date = VAL,
    **settings: Any,
) -> RunResult:
    """One ``Covered`` run, one scenario per shock (constant over the months; the file reaches
    month 60, so a run may set a longer horizon)."""
    shock = scenario_file("shock", 0, 60, len(shocks), lambda s, u: shocks[s])
    return run(
        compiled_covered(),
        model_points=points,
        scenarios={"shock": shock},
        settings=RunSettings(valuation_date=valuation_date, **settings),
        run_dir=tmp_path / name,
    )


STAGE_SCHEMA = Schema(
    fields=(
        Field("group_id", "string", unique=True),
        Field("j", "number", "Rate[Monthly]", ge=0, le=0.1),
    ),
    id_field="group_id",
)

MEMBERS = {"G0": ["B01", "A00", "A02"], "G1": ["A01", "B00", "B02"]}
"""Two groups across two class runs (``A..`` and ``B..``), members listed out of order."""


def stage_groups(ids: tuple[str, ...] = ("G1", "G0"), j: tuple[float, ...] = (0.004, 0.002)) -> Any:
    """The stage's model points: one row per group, listed out of order."""
    return pa.table(
        {"group_id": pa.array(list(ids), pa.string()), "j": pa.array(list(j), pa.float64())}
    )


class Release(Aggregation):
    """A toy CSM roll-forward (design section 9's ``CsmRollForward`` in miniature): a backward
    recursion over a group series (``cu_next``) feeds the forward ones (``csm``, ``lc``), whose
    inits are scalars formed from a scalar series and a fixed month of another."""

    model_point = STAGE_SCHEMA
    bases = (Basis("cur"),)
    series = (
        GroupSum("cu_g", "cur", "cu", "CoverageUnits"),
        GroupSum("fcf_g", "cur", "fcf", "Money", start=0),
        GroupSum("p_g", "cur", "premium_total", "Money", time=False),
    )

    @var
    def d(self) -> Factor:
        return 1.0 / (1 + self.group.j)

    @var(terminal=0)
    def cu_next(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t + 1) + self.d() * self.cu_next(t + 1)

    @var
    def units(self, t: Int) -> CoverageUnits:
        return self.series.cu_g(t) + self.cu_next(t)

    @var
    def csm_0(self) -> Money:
        gap = self.series.p_g() - self.series.fcf_g.at(0)
        return where(gap > 0, gap, 0.0)

    @var
    def lc_0(self) -> Money:
        gap = self.series.fcf_g.at(0) - self.series.p_g()
        return where(gap > 0, gap, 0.0)

    @var
    def acc(self, t: Int) -> Money:
        return self.group.j * self.csm(t - 1)

    @var
    def release(self, t: Int) -> Money:
        u = self.units(t)
        pre = self.csm(t - 1) + self.acc(t)
        return where(u > 0, pre * (self.series.cu_g(t) / where(u > 0, u, 1.0)), pre)

    @var(init=lambda g: g.csm_0())
    def csm(self, t: Int) -> Money:
        return (self.csm(t - 1) + self.acc(t)) - self.release(t)

    @var(init=lambda g: g.lc_0())
    def lc(self, t: Int) -> Money:
        f0 = self.series.fcf_g(t - 1)
        return where(f0 > 0, self.lc(t - 1) * (self.series.fcf_g(t) / where(f0 > 0, f0, 1.0)), 0.0)


def closed_form_release(
    cu_g: list[float], fcf_g: list[float], p_g: float, j: float
) -> dict[str, list[float]]:
    """What ``Release`` computes for one group and scenario, written independently with plain
    floats in the model's grouping, from the group series."""
    horizon = len(cu_g) - 1
    d = 1.0 / (1 + j)
    cu_next = [0.0] * (horizon + 1)
    for t in range(horizon - 1, -1, -1):
        cu_next[t] = cu_g[t + 1] + d * cu_next[t + 1]
    units = [0.0] + [cu_g[t] + cu_next[t] for t in range(1, horizon + 1)]
    gap = p_g - fcf_g[0]
    csm = [gap if gap > 0 else 0.0] + [0.0] * horizon
    lc = [-gap if -gap > 0 else 0.0] + [0.0] * horizon
    acc = [0.0] * (horizon + 1)
    release = [0.0] * (horizon + 1)
    for t in range(1, horizon + 1):
        acc[t] = j * csm[t - 1]
        pre = csm[t - 1] + acc[t]
        u = units[t]
        release[t] = pre * (cu_g[t] / u) if u > 0 else pre
        csm[t] = (csm[t - 1] + acc[t]) - release[t]
        f0 = fcf_g[t - 1]
        lc[t] = lc[t - 1] * (fcf_g[t] / f0) if f0 > 0 else 0.0
    return {
        "cu_next": cu_next,
        "units": units,
        "acc": acc,
        "release": release,
        "csm": csm,
        "lc": lc,
    }


def release_bases(
    tmp_path: Path, shocks: tuple[float, ...] = (1.0,), **settings: Any
) -> dict[str, Any]:
    """The ``cur`` basis of ``Release``: two class runs with disjoint policies."""
    a = run_covered(tmp_path, f"class-a-{len(shocks)}", covered_points("A"), shocks, **settings)
    b = run_covered(
        tmp_path, f"class-b-{len(shocks)}", covered_points("B", scale=1.5), shocks, **settings
    )
    return {"cur": [a, b]}


def run_stage(
    tmp_path: Path,
    stage: type[Aggregation],
    series: dict[str, Any],
    name: str = "stage",
    groups: Any = None,
    scenarios: dict[str, Any] | None = None,
    valuation_date: dt.date = VAL,
    **settings: Any,
) -> RunResult:
    return run(
        build(stage),
        model_points=stage_groups() if groups is None else groups,
        scenarios=scenarios,
        series=series,
        settings=RunSettings(valuation_date=valuation_date, **settings),
        run_dir=tmp_path / name,
    )


# ------------------------------------------------------------ aggregation runs (B2)

V_REC = dt.date(2025, 12, 31)
"""The toy group's recognition date (in-force 3.1's)."""

V_NOW = dt.date(2026, 12, 31)
"""A valuation date a year later: ``tau = tau0 = 12``."""

MEMBER_SCHEMA = Schema(
    fields=(
        int_field("policy_id", ge=1, unique=True),
        Field("portfolio", "string"),
        int_field("cohort", ge=1900, le=2200),
        Field("profit_bucket", "enum", enum=PROFIT_BUCKETS),
        Field("issue_date", "date"),
        Field("premium", "number", "MoneyPerPolicy", gt=0),
        Field("pols_if_init", "number", "Count", gt=0),
        int_field("rem_m", ge=1, le=1000),
    ),
    horizon="rem_m",
)

EXPENSE = Table("expense", keys=(), columns=(Column("maint", "MoneyPerPolicy", ge=0),))
ILP = Table("ilp", keys=(), columns=(Column("ilp", "Factor", ge=0, le=0.5),))
RA_PARAM = Table("ra_param", keys=(), columns=(Column("alpha", "Prob"),))


class Member(Product):
    """A projection of a group's members (design section 9's class runs in miniature): its
    model points carry the group key and the issue date; the discount curve, the CPI path and
    the illiquidity table are its financial inputs, the expense and RA tables its
    non-financial ones."""

    model_point = MEMBER_SCHEMA
    assumption_tables = (EXPENSE, ILP, RA_PARAM)
    scenario_inputs = (
        ScenarioInput("rf_df", "Factor", gt=0),
        ScenarioInput("cpi", "Index", gt=0),
    )
    outputs = ("x", "cf", "cu", "fcf", "premium_total")

    @var(init=lambda mp: mp.pols_if_init, role="in_force")
    def x(self, t: Int) -> Count:
        return self.x(t - 1) * 0.98

    @var
    def esc(self, t: Int) -> Factor:
        return self.scen.cpi(t - 1) / self.scen.cpi.at(0)

    @var
    def cf(self, t: Int) -> Money:
        return self.x(t - 1) * ((self.mp.premium * self.esc(t)) + self.tables.expense.maint)

    @var
    def cu(self, t: Int) -> CoverageUnits:
        return as_cu(self.x(t - 1) * self.mp.premium)

    @var(domain="state")
    def df(self, t: Int) -> Factor:
        return self.scen.rf_df(t) * (1 - self.tables.ilp.ilp)

    @var(terminal=0)
    def fcf(self, t: Int) -> Money:
        return (self.fcf(t + 1) + self.cf(t + 1)) * (self.df(t + 1) / self.df(t))

    @var
    def premium_total(self) -> Money:
        return (self.mp.pols_if_init * self.mp.premium) * 30.0


@functools.cache
def compiled_member() -> CompiledModel:
    return build(Member)


def member_points(
    ids: list[int],
    *,
    issue: dt.date = V_REC,
    cohort: int = 2025,
    bucket: str = "OTHER",
    rem_m: int = 24,
    scale: float = 1.0,
    cohorts: dict[int, int] | None = None,
    issues: dict[int, dt.date] | None = None,
) -> Any:
    """``MEMBER_SCHEMA`` model points of portfolio ``IA``; ``cohorts`` and ``issues`` override
    the cohort or the issue date of single policies."""
    cohorts = cohorts or {}
    issues = issues or {}
    return pa.table(
        {
            "policy_id": pa.array(ids, pa.int64()),
            "portfolio": pa.array(["IA"] * len(ids), pa.string()).dictionary_encode(),
            "cohort": pa.array([cohorts.get(i, cohort) for i in ids], pa.int16()),
            "profit_bucket": pa.array([bucket] * len(ids), pa.string()),
            "issue_date": pa.array([issues.get(i, issue) for i in ids], pa.date32()),
            "premium": pa.array([scale * (10.0 + 0.5 * (i % 7)) for i in ids], pa.float64()),
            "pols_if_init": pa.array([1.0 + 0.25 * (i % 5) for i in ids], pa.float64()),
            "rem_m": pa.array([rem_m] * len(ids), pa.int32()),
        }
    )


def keyless(column: str, value: float, **metadata: str) -> Any:
    """A one-row table with ``halley.<key>`` schema metadata."""
    table = pa.table({column: pa.array([value], pa.float64())})
    return table.replace_schema_metadata({f"halley.{k}": v for k, v in metadata.items()})


def _with_metadata(table: Any, meta: dict[str, str] | None) -> Any:
    return table.replace_schema_metadata({f"halley.{k}": v for k, v in (meta or {}).items()})


def curve(
    rate: float,
    n_scen: int = 1,
    column: str = "rf_df",
    u_hi: int = 60,
    meta: dict[str, str] | None = None,
) -> Any:
    """A discount curve ``exp(-rate * t / 12)`` (scenario ``s`` adds ``s / 100`` to the rate),
    months ``0..u_hi``, with ``halley.<key>`` schema metadata from ``meta``."""
    table = scenario_file(
        column, 0, u_hi, n_scen, lambda s, u: float(np.exp(-(rate + s / 100) * u / 12))
    )
    return _with_metadata(table, meta)


def cpi_path(
    growth: float, n_scen: int = 1, u_hi: int = 60, meta: dict[str, str] | None = None
) -> Any:
    """A CPI path ``100 * (1 + growth) ** (u / 12)``, months ``0..u_hi``."""
    table = scenario_file(
        "cpi", 0, u_hi, n_scen, lambda s, u: 100.0 * (1 + growth + s / 100) ** (u / 12)
    )
    return _with_metadata(table, meta)


def run_member(
    run_dir: Path,
    points: Any,
    *,
    rf_df: Any,
    cpi: Any,
    expense: Any,
    ilp: Any,
    ra_param: Any,
    valuation_date: dt.date,
    horizon: int,
    **settings: Any,
) -> RunResult:
    """One ``Member`` class run: every run of a basis takes the same horizon; ``settings`` adds
    to the ``RunSettings`` (a batch size, a number of workers)."""
    return run(
        compiled_member(),
        model_points=points,
        tables={"expense": expense, "ilp": ilp, "ra_param": ra_param},
        scenarios={"rf_df": rf_df, "cpi": cpi},
        settings=RunSettings(valuation_date=valuation_date, horizon=horizon, **settings),
        run_dir=run_dir,
    )


def group_row(**fields: Any) -> Any:
    """A groups file of one row (in-force 3.3): group ``IA-2025-OTHER``, recognised at
    ``V_REC``, with the stage's own field ``j``; ``fields`` adds or overrides columns (a value
    of ``None`` drops the column; a list gives several rows)."""
    row: dict[str, Any] = {
        "portfolio": "IA",
        "cohort": 2025,
        "profit_bucket": "OTHER",
        "recognition_date": V_REC,
        "j": 0.002,
    }
    row.update(fields)
    types = {"cohort": pa.int16(), "j": pa.float64()}
    columns: dict[str, Any] = {}
    for name, value in row.items():
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if isinstance(values[0], dt.date):
            columns[name] = pa.array(values, pa.date32())
        else:
            columns[name] = pa.array(values, types.get(name))
    return pa.table(columns)


RECOGNISE_OUTPUTS = ("csm", "lc", "release", "cu_next", "csm_0", "lc_0", "fcf_g", "p_g")
ROLL_OUTPUTS = (*RECOGNISE_OUTPUTS, "delta", "aoc", "lock", "months")


class Recognise(Release):
    """A toy recognition stage (design section 9's ``NewGroup`` in miniature) on the group
    schema of in-force 3.3: ``Release``'s roll-forward, storing the series it closes on so that
    a later in-force stage reads them from this run."""

    model_point = group_schema(Field("j", "number", "Rate[Monthly]", ge=0, le=0.1))
    outputs = RECOGNISE_OUTPUTS

    @var(domain="state")
    def fcf_g(self, t: Int) -> Money:
        return self.series.fcf_g(t)

    @var
    def p_g(self) -> Money:
        return self.series.p_g()


class Roll(Recognise):
    """A toy in-force stage (``InForceGroup`` in miniature): bases ``old``, ``new`` and ``cur``,
    the previous run's quantities as derived group fields, the locked-in curve ``p_l`` as its
    own scenario input, and the locked-in hashes on the groups file."""

    model_point = group_schema(
        Field("j", "number", "Rate[Monthly]", ge=0, le=0.1), previous=True, locked_in=True
    )
    bases = (Basis("old"), Basis("new"), Basis("cur"))
    series = (
        *Release.series[:2],
        GroupSum("fcf_old", "old", "fcf", "Money", start=0),
        GroupSum("fcf_new", "new", "fcf", "Money", start=0),
    )
    scenario_inputs = (ScenarioInput("p_l", "Factor", gt=0, column="rf_df"),)
    outputs = ROLL_OUTPUTS

    @var
    def delta(self) -> Money:
        return self.series.fcf_new.at(0) - self.series.fcf_old.at(0)

    @var
    def csm_0(self) -> Money:
        gap = (self.group.csm_prev - self.group.lc_prev) - self.delta()
        return where(gap > 0, gap, 0.0)

    @var
    def lc_0(self) -> Money:
        gap = self.delta() - (self.group.csm_prev - self.group.lc_prev)
        return where(gap > 0, gap, 0.0)

    @var
    def p_g(self) -> Money:
        return self.group.p_g_prev

    @var
    def aoc(self) -> Money:
        return (self.group.f_prev_tau - self.group.f_prev_0) + self.group.lc_prev_tau

    @var(domain="state")
    def lock(self, t: Int) -> Factor:
        return self.scen.p_l(t)

    @var
    def months(self) -> Int:
        return (self.group.tau0 - self.group.tau) + (self.group.i_acc + self.group.i_rec)
